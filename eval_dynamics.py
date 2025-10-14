from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze, FrozenDict
from models import Encoder, Decoder, Dynamics
from data import make_iterator
from pathlib import Path
from time import time
from utils import pack_mae_params, temporal_patchify, temporal_unpatchify, make_state, make_manager, try_restore, maybe_save, with_params
from einops import rearrange
import orbax.checkpoint as ocp

def pack_bottleneck_to_spatial(z_btLd, *, n_s: int, k: int):
    """
    (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
    Requires: N_b == n_s * k  (e.g., 512 -> 256 with k=2).
    """
    return rearrange(z_btLd, 'b t (n_s k) d -> b t n_s (k d)', n_s=n_s, k=k)


def init_models(rng, encoder, dynamics, patch_tokens, B, T, enc_n_latents, enc_d_bottleneck, packing_factor, num_spatial_tokens):
    rng, params_rng, mae_rng, dropout_rng = jax.random.split(rng, 4)

    enc_vars = encoder.init(
        {"params": params_rng, "mae": mae_rng, "dropout": dropout_rng},
        patch_tokens, deterministic=True
    )
    fake_enc_z = jnp.ones((B, T, enc_n_latents, enc_d_bottleneck), dtype=jnp.float32)
    fake_packed_z = pack_bottleneck_to_spatial(fake_enc_z, n_s=num_spatial_tokens, k=packing_factor)
    fake_actions = jnp.ones((B, T), dtype=jnp.int32)
    fake_step_idxs = jnp.full((B, T), 0, dtype=jnp.int32)
    fake_signal_idxs = jnp.full((B, T), 0, dtype=jnp.int32)
    dynamics_vars = dynamics.init(
        {"params": params_rng, "dropout": dropout_rng},
        fake_actions,
        fake_step_idxs,
        fake_signal_idxs,
        fake_packed_z,
    )
    return rng, enc_vars, dynamics_vars

def _read_only_tokenizer_meta(tokenizer_ckpt_dir: str):
    """Restore just the JSON meta from a tokenizer checkpoint directory."""
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    return latest, restored.meta


def load_pretrained_encoder_params(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    enc_vars,
    sample_patches_btnd,
):
    """Return enc_vars with restored encoder params from tokenizer ckpt (discard decoder)."""
    # -- (A) read meta only with a meta-only manager
    latest, meta = _read_only_tokenizer_meta(tokenizer_ckpt_dir)
    if "enc_kwargs" not in meta or "dec_kwargs" not in meta:
        raise ValueError("Tokenizer checkpoint meta missing enc_kwargs/dec_kwargs")
    enc_kwargs = meta["enc_kwargs"]
    dec_kwargs = meta["dec_kwargs"]

    # -- (B) build abstract trees that match the saved structure
    dec = Decoder(**dec_kwargs)
    rng_e1, rng_e2, rng_d1 = jax.random.split(rng, 3)

    B, T = sample_patches_btnd.shape[:2]
    n_lat = enc_kwargs["n_latents"]
    d_b   = enc_kwargs["d_bottleneck"]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)

    dec_vars = dec.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)

    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)

    state_example = make_state(
        params=packed_example,
        opt_state=opt_state_example,
        rng=rng_e2,
        step=0,
    )
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    # -- (C) now do a full restore (state + meta) using a manager with both items
    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        )
    )

    packed_params = restored.state["params"]        # {"enc": ..., "dec": ...}
    enc_params = packed_params["enc"]
    new_enc_vars = with_params(enc_vars, enc_params)
    return new_enc_vars, meta

def make_dynamics_meta(
    *,
    enc_kwargs: dict,
    dynamics_kwargs: dict,
    H: int, W: int, C: int,
    patch: int,
    k_max: int,
    packing_factor: int,
    n_s: int,
    tokenizer_ckpt_dir: str | None = None,
):
    return {
        "enc_kwargs": enc_kwargs,
        "dynamics_kwargs": dynamics_kwargs,
        "H": H, "W": W, "C": C, "patch": patch,
        "k_max": k_max,
        "packing_factor": packing_factor,
        "n_s": n_s,
        "tokenizer_ckpt_dir": tokenizer_ckpt_dir,
    }

@partial(jax.jit, static_argnames=("dynamics", "batch_size", "ctx_length", "num_sampling_steps", "enc_d_bottleneck", "n_s", "packing_factor", "k_max"))
def predict_one_step(
    dynamics: Dynamics,
    dynamics_vars: dict,
    z_bottleneck_ctx: jax.Array,
    actions_ctx: jax.Array,
    actions_curr: jax.Array,
    batch_size: int,
    ctx_length: int,
    num_sampling_steps: int,
    enc_d_bottleneck: int,
    n_s: int,
    packing_factor: int,
    k_max: int,
    rng: jax.random.PRNGKey,
) -> jax.Array:
    """
    Denoise the current frame's spatial tokens given context latents+actions.

    Args:
        dynamics:        Flax module (used via .apply)
        dynamics_vars:   Variables/params for `dynamics`
        z_bottleneck_ctx: (B, T_ctx, N_latents, D_b)
        actions_ctx:      (B, T_ctx, D_a)
        actions_curr:     (B, 1,     D_a)
        batch_size:       B (static for JIT)
        ctx_length:       T_ctx (static for JIT)
        num_sampling_steps: K diffusion steps; step size d = 1/K (static)
        n_s:              number of spatial tokens after packing (static)
        packing_factor:   spatial packing factor; D_s = D_b * packing_factor (static)
        k_max:            discretization for tau index in [0, k_max-1] (static)
        rng:              PRNGKey

    Returns:
        final_z_curr: (B, T_ctx+1, N_s, D_s)
    """
    # ---- RNG keys ----
    rng, ctx_noise_rng, curr_noise_rng, sampling_rng = jax.random.split(rng, 4)

    # ---- Pack context latents to spatial tokens & lightly noise ----
    # z_ctx: (B, T_ctx, N_s, D_s)
    z_ctx = pack_bottleneck_to_spatial(z_bottleneck_ctx, n_s=n_s, k=packing_factor)
    z_ctx = z_ctx + 0.1 * jax.random.normal(ctx_noise_rng, (batch_size, ctx_length, n_s, enc_d_bottleneck * packing_factor), dtype=z_ctx.dtype)

    # Add a fresh noisy slot for current timestep
    B = batch_size
    T_total = ctx_length + 1
    _, _, N_s, D_s = z_ctx.shape
    z0 = jax.random.normal(curr_noise_rng, (B, 1, N_s, D_s), dtype=z_ctx.dtype)
    z_curr = jnp.concatenate([z_ctx, z0], axis=1)  # (B, T_ctx+1, N_s, D_s)

    # ---- Actions concat: (B, T_ctx+1, D_a) ----
    actions = jnp.concatenate([actions_ctx, actions_curr], axis=1)

    # ---- Schedule tensors (constant over T) ----
    # Step size d = 1/K (float32)
    d = jnp.reciprocal(jnp.asarray(num_sampling_steps, dtype=jnp.float32))
    step_value = jnp.full((B, T_total), d, dtype=jnp.float32)

    # Step index e = log2(K) as int32, broadcast to (B, T_total)
    step_idx_scalar = jnp.round(jnp.log2(jnp.asarray(num_sampling_steps, dtype=jnp.float32)))
    step_idx_scalar = step_idx_scalar.astype(jnp.int32)
    step_idx = jnp.full((B, T_total), step_idx_scalar, dtype=jnp.int32)

    # Signal level τ starts at 0 everywhere
    signal_value = jnp.zeros((B, T_total), dtype=jnp.float32)
    signal_idx   = jnp.zeros((B, T_total), dtype=jnp.int32)

    # ---- One diffusion step (scanned K times) ----
    def sample_step(carry, n):
        """Single Euler diffusion step."""
        z_curr, signal_idx, signal_value = carry

        # Per-step dropout key
        drop_rng = jax.random.fold_in(sampling_rng, n)

        # Dynamics forward
        flow = dynamics.apply(
            dynamics_vars,
            actions,                # (B, T, D_a)
            step_idx,               # (B, T)
            signal_idx,             # (B, T)
            z_curr,                 # (B, T, N_s, D_s)
            rngs={"dropout": drop_rng},
            deterministic=True,
        )

        # Euler update: z <- z + flow * d
        z_curr = z_curr + flow * step_value[..., None, None]

        # τ <- τ + d; idx = floor(τ * k_max), clipped
        signal_value = signal_value + step_value
        new_signal_idx = jnp.floor(signal_value * jnp.asarray(k_max, jnp.float32)).astype(jnp.int32)
        new_signal_idx = jnp.clip(new_signal_idx, 0, k_max - 1)

        return (z_curr, new_signal_idx, signal_value), None

    (final_state, _) = jax.lax.scan(
        sample_step,
        (z_curr, signal_idx, signal_value),
        jnp.arange(num_sampling_steps),
    )
    final_z_curr, _, _ = final_state
    return final_z_curr

if __name__ == "__main__":
    log_dir = Path("./logs"); log_dir.mkdir(parents=True, exist_ok=True)
    run_name = "test_dynamics"
    run_dir = log_dir / run_name; run_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(0)
    # dataset parameters ...
    B, T, H, W, C = 8, 64, 32, 32, 3
    B_self = int(0.25 * B)
    pixels_per_step = 2 # how many pixels the agent moves per step
    size_min = 6 # minimum size of the square
    size_max = 14 # maximum size of the square
    hold_min = 4 # how long the agent holds a direction for
    hold_max = 9 # how long the agent holds a direction for

    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = 256


    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = 256

    next_batch = make_iterator(B, T, H, W, C, pixels_per_step, size_min, size_max, hold_min, hold_max)
    rng, batch_rng = jax.random.split(rng)
    rng, (frames, actions) = next_batch(rng)

    # ----- models -----
    enc_n_latents, enc_d_bottleneck = 16, 32
    enc_kwargs = {
        "d_model": 64, "n_latents": enc_n_latents, "n_patches": num_patches,
        "n_heads": 4, "depth": 8, "dropout": 0.0,
        "d_bottleneck": enc_d_bottleneck, "mae_p_min": 0.1, "mae_p_max": 0.1, "time_every": 4,
    }
    packing_factor = 2
    n_s = enc_n_latents // packing_factor
    dynamics_kwargs = {
        "d_model": 128, "n_s": n_s, "d_spatial": enc_d_bottleneck * packing_factor,
        "d_bottleneck": enc_d_bottleneck, "k_max": k_max, "n_r": 10,
        "n_heads": 4, "depth": 4, "dropout": 0.0
    }

    encoder = Encoder(**enc_kwargs)
    dynamics = Dynamics(**dynamics_kwargs)

    init_patches = temporal_patchify(frames, patch)
    rng, enc_vars, dynamics_vars = init_models(
        rng, encoder, dynamics, init_patches, B, T, enc_n_latents, enc_d_bottleneck,
        packing_factor, n_s
    )

    # ====== (A) Optional: load a pretrained encoder from tokenizer checkpoints ======
    # Set this path if you want to load a trained encoder; otherwise leave as None to use fresh init.
    TOKENIZER_CKPT_DIR = "/home/edward/projects/tiny_dreamer_4/logs/test/checkpoints" 
    if TOKENIZER_CKPT_DIR is not None:
        # use init_patches as sample to build shapes if needed
        enc_vars, tok_meta = load_pretrained_encoder_params(
            TOKENIZER_CKPT_DIR,
            rng=rng,
            encoder=encoder,
            enc_vars=enc_vars,
            sample_patches_btnd=init_patches,
        )
        print(f"[encoder] Restored pretrained encoder params from: {TOKENIZER_CKPT_DIR}")
    # ----- dynamics trainables -----
    params = dynamics_vars["params"]
    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)
    max_steps = 1_000_000

    # ====== (B) Orbax manager for dynamics run + try to restore ======
    ckpt_dir = (run_dir / "checkpoints")
    mngr = make_manager(ckpt_dir, max_to_keep=5, save_interval_steps=10_000)

    meta = make_dynamics_meta(
        enc_kwargs=enc_kwargs,
        dynamics_kwargs=dynamics_kwargs,
        H=H, W=W, C=C, patch=patch,
        k_max=k_max, packing_factor=packing_factor, n_s=n_s,
        tokenizer_ckpt_dir=TOKENIZER_CKPT_DIR
    )

    # Build example trees for safe restore
    state_example = make_state(params, opt_state, rng, step=0)
    restored = try_restore(mngr, state_example, meta)

    start_step = 0
    if restored is not None:
        latest_step, r = restored
        params     = r.state["params"]
        opt_state  = r.state["opt_state"]
        rng        = r.state["rng"]
        start_step = int(r.state["step"])
        dynamics_vars = with_params(dynamics_vars, params)
        print(f"[dynamics] Restored checkpoint step={latest_step} from {ckpt_dir}")

    # ====== (C) Visualize the loaded dynamics policy ======
    try:
        ctx_length = 8
        fake_z_bottleneck_ctx = jnp.zeros((B, ctx_length, enc_n_latents, enc_d_bottleneck), dtype=jnp.float32)
        fake_actions_ctx = jnp.zeros((B, ctx_length), dtype=jnp.int32)
        fake_actions_curr = jnp.zeros((B, 1), dtype=jnp.int32)
        num_sampling_steps = 4
        sampling_rng, rng = jax.random.split(rng)

        one_step_kwargs = {
            "dynamics": dynamics,
            "dynamics_vars": dynamics_vars,
            "z_bottleneck_ctx": fake_z_bottleneck_ctx,
            "actions_ctx": fake_actions_ctx,
            "actions_curr": fake_actions_curr,
            "batch_size": B,
            "ctx_length": ctx_length,
            "num_sampling_steps": num_sampling_steps,
            "enc_d_bottleneck": enc_d_bottleneck,
            "n_s": n_s,
            "packing_factor": packing_factor,
            "k_max": k_max,
            "rng": sampling_rng
        }
        for _ in range(10):
            start_time = time()
            predict_one_step(**one_step_kwargs)
            end_time = time()
            print(f"Time taken: {end_time - start_time} seconds")
        import ipdb; ipdb.set_trace()
    finally:
        mngr.wait_until_finished()
