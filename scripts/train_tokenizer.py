from dataclasses import dataclass, asdict
from functools import partial
from typing import Optional
import time
from pathlib import Path
import pprint
import jax
import jax.numpy as jnp
from jaxlpips import LPIPS
import imageio
import optax
import wandb
import pyrallis

from dreamer.models import Encoder, Decoder
from dreamer.data import make_iterator
from dreamer.utils import (
    temporal_patchify, temporal_unpatchify,
    make_state, make_manager, try_restore, maybe_save,
    pack_mae_params, unpack_mae_params
)
from configs.base import TokenizerConfig


def init_models(
    rng, encoder, decoder, patch_tokens, B, T, enc_n_latents, enc_d_bottleneck
):
    rng, params_rng, mae_rng, dropout_rng = jax.random.split(rng, 4)

    enc_vars = encoder.init(
        {"params": params_rng, "mae": mae_rng, "dropout": dropout_rng},
        patch_tokens, deterministic=True
    )
    fake_z = jnp.zeros((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": params_rng, "dropout": dropout_rng},
        fake_z, deterministic=True
    )
    return rng, enc_vars, dec_vars

# --- forward (no jit; we jit the train_step) ---
def forward_apply(
    encoder, decoder,
    enc_vars, dec_vars,
    patches_btnd,
    *,
    mae_key, drop_key, train: bool
):
    # Avoid TracerBool issues: pass a hardcoded bool OR use lax.cond
    rngs_enc = (
        {"mae": mae_key} if not train else {"mae": mae_key, "dropout": drop_key}
    )
    z_btLd, mae_info = encoder.apply(
        enc_vars, patches_btnd, rngs=rngs_enc, deterministic=not train
    )

    rngs_dec = {} if not train else {"dropout": drop_key}
    pred_btnd = decoder.apply(
        dec_vars, z_btLd, rngs=rngs_dec, deterministic=not train
    )
    return pred_btnd, mae_info  # mae_info = (mae_mask, keep_prob)

# --- loss ---
def recon_loss_from_mae(pred_btnd, patches_btnd, mae_mask):
    masked_pred   = jnp.where(mae_mask, pred_btnd, 0.0)
    masked_target = jnp.where(mae_mask, patches_btnd, 0.0)
    num = jnp.maximum(mae_mask.sum(), 1)
    sq_err = jnp.sum((masked_pred - masked_target) ** 2)
    return sq_err / (num * pred_btnd.shape[-1])


def lpips_on_mae_recon(
    pred, target, mae_mask,
    *,
    H, W, C, patch,
    subsample_frac: float = 1.0, lpips_batch_size: int = 8, lpips_loss_fn
):
    """
    pred:    (B,T,Np,D)
    target:  (B,T,Np,D)
    mae_mask:     (B,T,Np,1)  True where patch is masked (must reconstruct)
    Returns scalar LPIPS averaged over (B,T).
    """
    # 1) Blend GT for visible patches => "recon_masked"
    recon_masked_btnd = jnp.where(mae_mask, pred, target)

    # 2) Unpatchify to (B,T,H,W,C) in [0,1]
    recon_imgs = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    target_imgs = temporal_unpatchify(target, H, W, C, patch)

    # 3) Optional subsample frames over T to save compute
    if subsample_frac < 1.0:
        B, T = recon_imgs.shape[:2]
        step = max(1, int(1.0/subsample_frac))
        idx = jnp.arange(T)[::step]
        recon_imgs = recon_imgs[:, idx]
        target_imgs = target_imgs[:, idx]

    # 4) Rescale to [-1,1] for LPIPS
    recon_lp = jnp.clip(recon_imgs * 2.0 - 1.0, -1.0, 1.0)
    target_lp = jnp.clip(target_imgs * 2.0 - 1.0, -1.0, 1.0)

    # 5) Flatten B,T for a single LPIPS call: (B*T,H,W,C)
    BT = recon_lp.shape[0] * recon_lp.shape[1]
    H_, W_, C_ = recon_lp.shape[2], recon_lp.shape[3], recon_lp.shape[4]
    recon_lp = recon_lp.reshape((BT, H_, W_, C_))
    target_lp = target_lp.reshape((BT, H_, W_, C_))

    # 6) LPIPS returns per-example loss; average it (in mini-batches to avoid OOM)
    lp_all = []
    for i in range(0, BT, lpips_batch_size):
        end_i = min(i + lpips_batch_size, BT)
        lp_batch = lpips_loss_fn(recon_lp[i:end_i], target_lp[i:end_i])
        lp_all.append(lp_batch)
    lp = jnp.concatenate(lp_all, axis=0)
    return jnp.mean(lp)

# --- viz step ---
@partial(jax.jit, static_argnames=("encoder","decoder","patch"))
def viz_step(
    encoder, decoder, enc_vars, dec_vars, batch, *, patch, mae_key, drop_key
):
    # Same preprocessing as train
    patches_btnd = temporal_patchify(batch, patch)  # (B,T,Np,D)

    # Run full model (no dropout during viz)
    pred_btnd, (mae_mask_btNp1, keep_prob_bt1) = forward_apply(
        encoder, decoder, enc_vars, dec_vars, patches_btnd,
        mae_key=mae_key, drop_key=drop_key, train=False
    )

    # Compose standard MAE visualization:
    # - masked_input: what the model actually sees (masked patches)
    # - recon_masked: inpaint only masked patches (visible patches kept as GT)
    masked_input_btnd  = jnp.where(mae_mask_btNp1, 0.0, patches_btnd)
    recon_masked_btnd  = jnp.where(mae_mask_btNp1, pred_btnd, patches_btnd)
    recon_full_btnd    = pred_btnd  # decoder everywhere

    return {
        "target": patches_btnd,
        "masked_input": masked_input_btnd,
        "recon_masked": recon_masked_btnd,
        "recon_full": recon_full_btnd,
        "mae_mask": mae_mask_btNp1,
        "keep_prob": keep_prob_bt1,
    }


# --- train step ---
@partial(
    jax.jit,
    static_argnames=("cfg", "encoder", "decoder", "tx", "lpips_fn")
)
def train_step(
    cfg: TokenizerConfig,
    encoder, decoder, tx, lpips_fn, params, opt_state, enc_vars, dec_vars, batch,
    *,
    master_key, step,
):
    """
    (master_key, params, opt_state, model_state, batch)
        │
        ▼
    [ compute grads ]
        │
        ▼
    Optax: (grads, opt_state, params) → (updates, new_opt_state)
    Flax:  params ⟶ apply updates → new_params
        │
        ▼
    return (new_params, new_opt_state, new_model_state, metrics)
    """
    # 1) Prepare data
    patches_btnd = temporal_patchify(batch, cfg.patch)  # (B, T, Np, Dp)

    # 2) Make per-step RNGs (fold_in ensures different masks per step even if base key repeats)
    step_key  = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    # 3) Define loss fn (closes over encoder/decoder + non-param states)
    def loss_fn(packed_params):
        # Replace params in vars
        ev, dv = unpack_mae_params(packed_params, enc_vars, dec_vars)
        pred, mae_info = forward_apply(
            encoder, decoder, ev, dv, patches_btnd,
            mae_key=mae_key, drop_key=drop_key, train=True
        )
        mae_mask, keep_prob = mae_info
        mse = recon_loss_from_mae(pred, patches_btnd, mae_mask)

        # LPIPS on recon_masked vs target (unpatchified frames)
        if cfg.lpips_weight > 0.0:
            lpips_frac = 0.5

            lpips = lpips_on_mae_recon(
                pred, patches_btnd, mae_mask,
                H=cfg.env.H, W=cfg.env.W, C=cfg.env.C, patch=cfg.patch,
                subsample_frac=lpips_frac,
                lpips_batch_size=cfg.lpips_batch_size,
                lpips_loss_fn=lpips_fn,
            )
            total = mse + cfg.lpips_weight * lpips
        else:
            lpips = 0.0
            total = mse

        aux = {
            "loss_total": total,
            "loss_mse": mse,
            "loss_lpips": lpips,
            "keep_prob": keep_prob,
        }

        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # 4) Update
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 5) Put params back into variables for next step
    new_enc_vars, new_dec_vars = unpack_mae_params(
        new_params, enc_vars, dec_vars
    )
    return new_params, opt_state, new_enc_vars, new_dec_vars, aux


@pyrallis.wrap()
def run(cfg: TokenizerConfig):
    pprint.pprint(cfg)

    # wandb init
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            project=cfg.wandb_project,
            name=cfg.run_name,
            group=cfg.wandb_group or cfg.run_name,
            dir=str(Path(cfg.log_dir).resolve()),
        )
        print(f"[wandb] Initialized run: {wandb.run.name if wandb.run else 'N/A'}")

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir = log_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(cfg.seed)

    # data
    num_patches = (cfg.env.H // cfg.patch) * (cfg.env.W // cfg.patch)
    D_patch = cfg.patch * cfg.patch * cfg.env.C

    if cfg.env.env_type == "TinyEnv":
        _next_batch = make_iterator(
            cfg.env.B, cfg.env.T, cfg.env.H, cfg.env.W, cfg.env.C,
            cfg.env.pixels_per_step,
            cfg.env.size_min, cfg.env.size_max,
            cfg.env.hold_min, cfg.env.hold_max,
            fg_min_color=0 if cfg.env.diversify_data else 128,
            fg_max_color=255 if cfg.env.diversify_data else 128,
            bg_min_color=0 if cfg.env.diversify_data else 255,
            bg_max_color=255 if cfg.env.diversify_data else 255,
        )
    elif cfg.env.env_type == "Atari":
        raise NotImplementedError("not implemented yet.")

    def next_batch(rng):
        rng, (videos, actions, rewards) = _next_batch(rng)
        return rng, videos

    def unpatchify_outputs(out):
        return jnp.concatenate(
            temporal_unpatchify(
                out, cfg.env.H, cfg.env.W, cfg.env.C, cfg.patch
            ).squeeze(),
            axis=1,
        )

    rng, batch_rng = jax.random.split(rng)
    rng, first_batch = next_batch(batch_rng)  # warmup

    # models
    print("Initializing models...")
    enc_kwargs = {
        "d_model": cfg.d_model, "n_latents": cfg.enc_n_latents,
        "n_patches": num_patches, "n_heads": cfg.n_heads,
        "depth": cfg.enc_depth, "dropout": cfg.enc_dropout,
        "d_bottleneck": cfg.enc_d_bottleneck,
        "mae_p_min": 0.0, "mae_p_max": 0.9, "time_every": 4,
    }
    dec_kwargs = {
        "d_model": cfg.d_model, "n_latents": cfg.enc_n_latents,
        "n_patches": num_patches, "n_heads": cfg.n_heads,
        "depth": cfg.dec_depth, "d_patch": D_patch, "dropout": cfg.dec_dropout,
        "time_every": 4,
    }
    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)

    first_patches = temporal_patchify(first_batch, cfg.patch)
    rng, enc_vars, dec_vars = init_models(
        rng, encoder, decoder, first_patches,
        cfg.env.B, cfg.env.T, cfg.enc_n_latents, cfg.enc_d_bottleneck,
    )

    # LPIPS initialization (once)
    print("Initializing LPIPS...")
    lpips_fn = (
        LPIPS(pretrained_network="alexnet")  # or "vgg", "squeeze"
        if cfg.lpips_weight > 0.0 else None
    )

    # optim
    params = pack_mae_params(enc_vars, dec_vars)
    tx = optax.adamw(cfg.lr)
    opt_state = tx.init(params)

    # ---------- ORBAX: manager + (optional) restore ----------
    print("Setting up checkpoint manager...")
    ckpt_dir = run_dir / "checkpoints"
    mngr = make_manager(
        ckpt_dir,
        max_to_keep=cfg.ckpt_max_to_keep,
        save_interval_steps=cfg.ckpt_save_every,
    )

    state_example = make_state(params, opt_state, rng, step=0)
    meta_example = {
        "enc_kwargs": enc_kwargs, "dec_kwargs": dec_kwargs,
        "H": cfg.env.H, "W": cfg.env.W, "C": cfg.env.C, "patch": cfg.patch
    }

    restored = try_restore(mngr, state_example, meta_example)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        # Unpack state back to your locals
        params = r.state["params"]
        opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        # Optional: you can read r.meta here if you want to sanity-check config.

        # Rebuild enc_vars/dec_vars from params so downstream apply() uses the restored params.
        enc_vars, dec_vars = unpack_mae_params(params, enc_vars, dec_vars)
        print(f"[restore] Resumed from {ckpt_dir} at step={latest_step}")

    # ---------- Train loop ----------
    start_wall = time.time()
    try:
        for step in range(start_step, cfg.max_steps):
            data_start_t = time.time()
            rng, batch = next_batch(rng)
            data_t = time.time() - data_start_t

            train_start_t = time.time()
            rng, master_key = jax.random.split(rng)
            params, opt_state, enc_vars, dec_vars, aux = train_step(
                cfg,
                encoder, decoder, tx, lpips_fn, params, opt_state,
                enc_vars, dec_vars, batch,
                master_key=master_key, step=step,
            )
            train_t = time.time() - train_start_t

            # Log
            if step % cfg.log_every == 0:
                mse_loss = float(aux['loss_mse'])
                lpips_loss = float(aux['loss_lpips'])
                total_loss = float(aux['loss_total'])
                psnr = float(10 * jnp.log10(1.0 / jnp.maximum(mse_loss, 1e-10)))
                step_t = data_t + train_t
                total_t = time.time() - start_wall

                pieces = [
                    f"[train] step={step:06d}",
                    f"total={total_loss:.6f}",
                    f"rmse={jnp.sqrt(mse_loss):.6f}",
                    f"lpips={lpips_loss:.4f}",
                    f"psnr={psnr:.2f}",
                    f"t={step_t:.3f}s",
                    f"total_t={total_t:.1f}s",
                ]
                print(" | ".join(pieces))

                if cfg.use_wandb and wandb.run is not None:
                    wandb.log({
                        "tokenizer/train/loss_total": total_loss,
                        "tokenizer/train/loss_mse": mse_loss,
                        "tokenizer/train/loss_lpips": lpips_loss,
                        "tokenizer/train/psnr": psnr,
                        "tokenizer/train/step_time": step_t,
                        "tokenizer/train/total_time": total_t,
                    }, step=step)

            # Save (async)
            state = make_state(params, opt_state, rng, step)
            maybe_save(mngr, step, state, meta_example)

            # Viz
            if cfg.viz_every and (step % cfg.viz_every == 0):
                rng, viz_key = jax.random.split(rng)
                mae_key, drop_key, vis_batch_key = jax.random.split(viz_key, 3)
                _, viz_batch = next_batch(vis_batch_key)
                viz_batch = viz_batch[:8, :1]
                out = viz_step(
                    encoder, decoder, enc_vars, dec_vars, viz_batch,
                    patch=cfg.patch, mae_key=mae_key, drop_key=drop_key
                )
                target = unpatchify_outputs(out["target"])
                masked_in = unpatchify_outputs(out["masked_input"])
                rec_masked  = unpatchify_outputs(out["recon_masked"])
                rec_unmasked  = unpatchify_outputs(out["recon_full"])
                grid = jnp.concatenate(
                    [target, masked_in, rec_masked, rec_unmasked]
                )
                grid = jnp.asarray(grid * 255.0, dtype=jnp.uint8)
                img_path = run_dir / f"step_{step:06d}.png"
                imageio.imwrite(img_path, grid)

                if cfg.use_wandb and wandb.run is not None:
                    wandb.log({
                        "tokenizer/viz/reconstruction": wandb.Image(str(img_path)),
                    }, step=step)
    finally:
        mngr.wait_until_finished()
        if cfg.use_wandb and wandb.run is not None:
            wandb.finish()
            print("[wandb] Finished logging.")


if __name__ == "__main__":
    run()