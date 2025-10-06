from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze, FrozenDict
from tokenizer import Encoder, Decoder
from data import make_iterator, patchify, unpatchify
import imageio

# --- helpers ---
temporal_patchify = jax.jit(
    jax.vmap(patchify, in_axes=(1, None), out_axes=1),  # (B,T,H,W,C) -> (B,T,Np,Dp)
    static_argnames=("patch",),
)

temporal_unpatchify = jax.jit(
    jax.vmap(unpatchify, in_axes=(1, None, None, None, None), out_axes=1),
    static_argnames=("H", "W", "C", "patch"),
)


def make_models(num_patches, D_patch, enc_n_latents=2, enc_d_bottleneck=3):
    encoder = Encoder(d_model=8, n_latents=enc_n_latents, n_heads=2, depth=2,
                      dropout=0, d_bottleneck=enc_d_bottleneck, mae_p_min=0.0, mae_p_max=0.9)
    decoder = Decoder(d_model=8, n_heads=2, depth=2, n_patches=num_patches,
                      d_patch=D_patch, dropout=0)
    return encoder, decoder

def init_models(rng, encoder, decoder, patch_tokens, B, T, enc_n_latents, enc_d_bottleneck):
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

# Pack params so we can optimize both modules with one optimizer.
def pack_params(enc_vars, dec_vars):
    return FrozenDict({
        "enc": enc_vars["params"],
        "dec": dec_vars["params"],
    })

def _with_params(variables, new_params):
    # works whether `variables` is a FrozenDict or a plain dict
    d = unfreeze(variables) if isinstance(variables, FrozenDict) else dict(variables)
    d["params"] = new_params
    return freeze(d)

def unpack_params(packed_params, enc_vars, dec_vars):
    enc_vars = _with_params(enc_vars, packed_params["enc"])
    dec_vars = _with_params(dec_vars, packed_params["dec"])
    return enc_vars, dec_vars


# --- forward (no jit; we jit the train_step) ---
def forward_apply(encoder, decoder, enc_vars, dec_vars, patches_btnd, *, mae_key, drop_key, train: bool):
    # Avoid TracerBool issues: pass a python bool here OR replace with lax.cond if needed.
    rngs_enc = {"mae": mae_key} if not train else {"mae": mae_key, "dropout": drop_key}
    z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd, rngs=rngs_enc, deterministic=not train)

    rngs_dec = {} if not train else {"dropout": drop_key}
    pred_btnd = decoder.apply(dec_vars, z_btLd, rngs=rngs_dec, deterministic=not train)
    return pred_btnd, mae_info  # mae_info = (mae_mask, keep_prob)

# --- loss ---
def recon_loss_from_mae(pred_btnd, patches_btnd, mae_mask):
    masked_pred   = jnp.where(mae_mask, pred_btnd, 0.0)
    masked_target = jnp.where(mae_mask, patches_btnd, 0.0)
    num = jnp.maximum(mae_mask.sum(), 1)
    return jnp.sum((masked_pred - masked_target) ** 2) / num

# --- viz step ---
@partial(jax.jit, static_argnames=("encoder","decoder","patch"))
def viz_step(encoder, decoder, enc_vars, dec_vars, batch, *, patch, mae_key, drop_key):
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
@partial(jax.jit, static_argnames=("encoder","decoder","tx","patch"))
def train_step(encoder, decoder, tx, params, opt_state, enc_vars, dec_vars, batch, *, patch, master_key, step):
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
    patches_btnd = temporal_patchify(batch, patch)  # (B,T,Np,Dp)

    # 2) Make per-step RNGs (fold_in ensures different masks per step even if base key repeats)
    step_key  = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    # 3) Define loss fn (closes over encoder/decoder + non-param states)
    def loss_fn(packed_params):
        # Replace params in vars
        ev, dv = unpack_params(packed_params, enc_vars, dec_vars)
        pred, mae_info = forward_apply(encoder, decoder, ev, dv, patches_btnd,
                                       mae_key=mae_key, drop_key=drop_key, train=True)
        mae_mask, keep_prob = mae_info
        loss = recon_loss_from_mae(pred, patches_btnd, mae_mask)
        return loss, {"loss": loss, "keep_prob": keep_prob}

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # 4) Update
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 5) Put params back into variables for next step
    new_enc_vars, new_dec_vars = unpack_params(new_params, enc_vars, dec_vars)
    return new_params, opt_state, new_enc_vars, new_dec_vars, aux

# --- usage example ---
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    B, T, H, W, C, square_size = 2, 10, 8, 8, 3, 4
    patch = 2
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C

    # data
    next_batch = make_iterator(B, T, H, W, C, square_size)
    rng, batch_rng = jax.random.split(rng)
    rng, first_batch = next_batch(rng)  # warmup

    # 1) Fix a tiny viz batch once
    viz_rng = jax.random.PRNGKey(0)
    _, viz_batch = next_batch(viz_rng)   # shape (B,T,H,W,C)
    viz_batch = viz_batch[:8, :1]        # just (8,1,H,W,C) to keep it tiny


    # models
    enc_n_latents, enc_d_bottleneck = 2, 3
    encoder, decoder = make_models(num_patches, D_patch, enc_n_latents, enc_d_bottleneck)
    first_patches = temporal_patchify(first_batch, patch)
    rng, enc_vars, dec_vars = init_models(rng, encoder, decoder, first_patches, B, T, enc_n_latents, enc_d_bottleneck)

    # optim
    params = pack_params(enc_vars, dec_vars)
    tx = optax.adamw(3e-4)
    opt_state = tx.init(params)

    # train loop
    for step in range(100000):
        # for now, fix rng to a single seed to debug.
        data_rng = jax.random.PRNGKey(0)
        _, batch = next_batch(data_rng)
        # rng, batch = next_batch(rng)
        rng, master_key = jax.random.split(rng)
        params, opt_state, enc_vars, dec_vars, aux = train_step(
            encoder, decoder, tx, params, opt_state, enc_vars, dec_vars, batch,
            patch=patch, master_key=master_key, step=step
        )
        mse_loss = float(aux['loss'])
        rmse_loss = jnp.sqrt(mse_loss)
        print(f"step {step:03d} | rmse loss={rmse_loss:.6f} | keep_prob≈{float(jnp.mean(aux['keep_prob'])):.3f}")
        if step % 10000 == 0:
            rng, viz_key = jax.random.split(rng)
            mae_key, drop_key = jax.random.split(viz_key)
            out = viz_step(encoder, decoder, enc_vars, dec_vars, viz_batch,
                        patch=patch, mae_key=mae_key, drop_key=drop_key)

            target = temporal_unpatchify(out["target"], H, W, C, patch).squeeze()
            masked_in = temporal_unpatchify(out["masked_input"], H, W, C, patch).squeeze()
            rec_masked  = temporal_unpatchify(out["recon_masked"], H, W, C, patch).squeeze()
            rec_unmasked  = temporal_unpatchify(out["recon_full"], H, W, C, patch).squeeze()

            # first stack the batches
            grid = jnp.concatenate([target, masked_in, rec_masked, rec_unmasked])
            grid = jnp.asarray(jnp.concatenate(grid, axis=1) * 255.0, dtype=jnp.uint8)
            imageio.imwrite(f"step_{step:03d}.png", grid)