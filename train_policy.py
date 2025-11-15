# train_policy.py
# Reinforcement learning training in the world model.
# Loads pretrained encoder, decoder, dynamics, reward head, and BC policy head.
# Trains a new policy head and value head on imagined rollouts.
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
from functools import partial
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from models import Encoder, Decoder, Dynamics, TaskEmbedder, PolicyHeadMTP, RewardHeadMTP, ValueHead
from data import make_iterator
from utils import (
    temporal_patchify,
    pack_bottleneck_to_spatial,
    with_params,
    make_state, make_manager, try_restore, maybe_save,
    pack_mae_params,
)

# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class RLConfig:
    # IO / ckpt
    run_name: str
    bc_rew_ckpt: str  # checkpoint from train_bc_rew_heads.py (contains dynamics, task_embedder, policy_head, reward_head)
    log_dir: str = "./logs"
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None  # if None, uses default entity
    wandb_project: str | None = None  # if None, uses run_name as project

    # data
    B: int = 64
    T: int = 64
    H: int = 32
    W: int = 32
    C: int = 3
    pixels_per_step: int = 2
    size_min: int = 6
    size_max: int = 14
    hold_min: int = 4
    hold_max: int = 9
    diversify_data: bool = True
    action_dim: int = 5  # number of categorical actions

    # tokenizer / dynamics config (should match BC/rew training)
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dyn: int = 128
    enc_depth: int = 8
    dec_depth: int = 8
    dyn_depth: int = 8
    n_heads: int = 4
    packing_factor: int = 2
    n_register: int = 4  # number of register tokens for dynamics
    n_agent: int = 1  # number of agent tokens for dynamics
    agent_space_mode: str = "wm_agent"

    # schedule
    k_max: int = 8

    # train
    max_steps: int = 1_000_000_000
    log_every: int = 5_000
    lr: float = 3e-4

    # eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely

    # RL-specific settings
    L: int = 2  # predict next L actions/rewards (from BC/rew training)
    num_reward_bins: int = 101  # twohot bins for symexp rewards
    num_value_bins: int = 101  # twohot bins for symexp values
    n_tasks: int = 128  # task-ID space for TaskEmbedder
    use_task_ids: bool = True  # True: discrete task IDs; False: vector embed
    
    # RL hyperparameters
    gamma: float = 0.997  # discount factor
    lambda_: float = 0.95  # lambda for TD(lambda) returns
    horizon: int = 32  # imagination horizon for rollouts
    context_length: int = 16  # length of context sequences sampled from videos

# ---------------------------
# Small helpers
# ---------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------
# Context sampling from videos
# ---------------------------

@partial(
    jax.jit,
    static_argnames=("context_length", "H", "W", "C"),
)
def sample_contexts(
    videos: jnp.ndarray,  # (B, T, H, W, C)
    actions: jnp.ndarray,  # (B, T)
    rewards: jnp.ndarray,  # (B, T)
    rng: jnp.ndarray,
    context_length: int,
    H: int,
    W: int,
    C: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample valid contexts from videos.
    
    For each video in the batch, samples a random start index from [0, T - context_length + 1)
    and extracts a context of length context_length. This ensures the furthest future valid
    context includes the last timestep's frame (index T-1) as its last element.
    
    Args:
        videos: (B, T, H, W, C) video frames
        actions: (B, T) action sequences
        rewards: (B, T) reward sequences
        rng: PRNGKey for sampling
        context_length: length of context sequences to extract
        H, W, C: spatial dimensions (static for JIT)
        
    Returns:
        context_frames: (B, context_length, H, W, C) sampled context frames
        context_actions: (B, context_length) sampled context actions
        context_rewards: (B, context_length) sampled context rewards
    """
    B, T = videos.shape[:2]
    
    # Valid start indices: [0, T - context_length + 1)
    # This ensures the context starting at (T - context_length) includes frame (T-1) as its last element
    max_start = T - context_length + 1
    
    # Sample a random start index per video
    start_indices = jax.random.randint(
        rng, (B,), minval=0, maxval=max_start, dtype=jnp.int32
    )  # (B,)
    
    # Extract contexts using dynamic slicing (JAX-compatible)
    # For each batch item b, extract frames[start_indices[b]:start_indices[b] + context_length]
    def extract_context_frames(video_seq, start_idx):
        """Extract context frames of length context_length starting at start_idx."""
        # video_seq: (T, H, W, C)
        # jax.lax.dynamic_slice: (operand, start_indices, slice_sizes)
        return jax.lax.dynamic_slice(
            video_seq,
            start_indices=(start_idx, 0, 0, 0),
            slice_sizes=(context_length, H, W, C)
        )
    
    def extract_context_1d(seq, start_idx):
        """Extract 1D context of length context_length starting at start_idx."""
        # seq: (T,)
        return jax.lax.dynamic_slice(
            seq,
            start_indices=(start_idx,),
            slice_sizes=(context_length,)
        )
    
    # Vectorize over batch dimension
    context_frames = jax.vmap(extract_context_frames, in_axes=(0, 0))(videos, start_indices)
    context_actions = jax.vmap(extract_context_1d, in_axes=(0, 0))(actions, start_indices)
    context_rewards = jax.vmap(extract_context_1d, in_axes=(0, 0))(rewards, start_indices)
    
    return context_frames, context_actions, context_rewards

# ---------------------------
# Checkpoint loading
# ---------------------------

def load_pretrained_tokenizer(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    decoder: Decoder,
    enc_vars,
    dec_vars,
    sample_patches_btnd,
):
    """Load pretrained encoder/decoder from tokenizer checkpoint."""
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]

    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    packed_params = restored.state["params"]
    enc_params = packed_params["enc"]
    dec_params = packed_params["dec"]
    new_enc_vars = with_params(enc_vars, enc_params)
    new_dec_vars = with_params(dec_vars, dec_params)
    print(f"[tokenizer] Restored encoder/decoder from {tokenizer_ckpt_dir} (step {latest})")
    return new_enc_vars, new_dec_vars, meta

def load_bc_rew_checkpoint(
    bc_rew_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_head_bc: PolicyHeadMTP,
    reward_head: RewardHeadMTP,
    dyn_vars,
    task_vars,
    pi_bc_vars,
    rew_vars,
    sample_actions: jnp.ndarray,
    sample_z1: jnp.ndarray,
):
    """
    Load pretrained dynamics, task_embedder, BC policy head, and reward head from BC/rew checkpoint.
    
    Returns:
        dyn_vars, task_vars, pi_bc_vars, rew_vars (all with loaded params)
        meta dict from checkpoint
    """
    # Create example state for checkpoint restoration
    params_example = {
        "dyn": dyn_vars["params"],
        "task": task_vars["params"],
        "pi": pi_bc_vars["params"],
        "rew": rew_vars["params"],
    }
    tx_dummy = optax.adam(1e-3)
    opt_state_example = tx_dummy.init(params_example)
    state_example = make_state(params_example, opt_state_example, rng, step=0)
    
    mngr = make_manager(bc_rew_ckpt_dir, item_names=("state", "meta"))
    restored = try_restore(mngr, state_example, meta_example={})
    if restored is None:
        raise FileNotFoundError(f"No BC/rew checkpoint found in {bc_rew_ckpt_dir}")
    
    latest_step, r = restored
    loaded_params = r.state["params"]
    
    # Extract params for each module
    dyn_params = loaded_params["dyn"]
    task_params = loaded_params["task"]
    pi_bc_params = loaded_params["pi"]
    rew_params = loaded_params["rew"]
    
    # Bind params into variables
    dyn_vars_loaded = with_params(dyn_vars, dyn_params)
    task_vars_loaded = with_params(task_vars, task_params)
    pi_bc_vars_loaded = with_params(pi_bc_vars, pi_bc_params)
    rew_vars_loaded = with_params(rew_vars, rew_params)
    
    meta = r.meta if hasattr(r, 'meta') and r.meta is not None else {}
    
    print(f"[bc_rew] Restored dynamics/task/policy_bc/reward from {bc_rew_ckpt_dir} (step {latest_step})")
    return dyn_vars_loaded, task_vars_loaded, pi_bc_vars_loaded, rew_vars_loaded, meta

# ---------------------------
# Training state dataclass
# ---------------------------

@dataclass
class TrainState:
    """Container for all training-related state (models, variables, optimizer, etc.)."""
    # Frozen models (loaded from checkpoints, not trained)
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    task_embedder: TaskEmbedder
    policy_head_bc: PolicyHeadMTP  # BC policy head (frozen, used as behavioral prior)
    reward_head: RewardHeadMTP  # Reward head (frozen)
    
    # Trainable models
    policy_head: PolicyHeadMTP  # RL policy head (trainable)
    value_head: ValueHead  # Value head (trainable)

    # vars/collections (frozen modules)
    enc_vars: dict
    dec_vars: dict
    dyn_vars: dict
    task_vars: dict
    pi_bc_vars: dict  # BC policy head vars (frozen)
    rew_vars: dict  # Reward head vars (frozen)

    # vars/collections (trainable modules)
    pi_vars: dict  # RL policy head vars
    val_vars: dict  # Value head vars

    # params packed for a single optimizer (subtrees: pi/val only)
    params: dict
    enc_kwargs: dict
    dec_kwargs: dict
    dyn_kwargs: dict
    tx: optax.Transform
    opt_state: optax.OptState
    mae_eval_key: jnp.ndarray

# ---------------------------
# Model initialization
# ---------------------------

def initialize_models(
    cfg: RLConfig,
    frames_init: jnp.ndarray,
    actions_init: jnp.ndarray,
) -> TrainState:
    """
    Initialize all models:
    - Load encoder/decoder from tokenizer checkpoint (from BC/rew meta)
    - Load dynamics, task_embedder, BC policy head, reward head from BC/rew checkpoint
    - Initialize new RL policy head and value head
    """
    patch = cfg.patch
    num_patches = (cfg.H // patch) * (cfg.W // patch)
    D_patch = patch * patch * cfg.C
    k_max = cfg.k_max

    # Encoder/decoder kwargs (should match BC/rew training)
    enc_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        n_heads=cfg.n_heads,
        depth=cfg.enc_depth,
        dropout=0.0,
        d_bottleneck=cfg.enc_d_bottleneck,
        mae_p_min=0.0, mae_p_max=0.0,
        time_every=4, latents_only_time=True,
    )
    dec_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_heads=cfg.n_heads,
        depth=cfg.dec_depth,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        d_patch=D_patch,
        dropout=0.0,
        mlp_ratio=4.0, time_every=4, latents_only_time=True,
    )
    n_spatial = cfg.enc_n_latents // cfg.packing_factor
    dyn_kwargs = dict(
        d_model=cfg.d_model_dyn,
        d_bottleneck=cfg.enc_d_bottleneck,
        d_spatial=cfg.enc_d_bottleneck * cfg.packing_factor,
        n_spatial=n_spatial, n_register=cfg.n_register,
        n_heads=cfg.n_heads, depth=cfg.dyn_depth,
        space_mode=cfg.agent_space_mode, n_agent=cfg.n_agent,
        dropout=0.0, k_max=k_max,
        time_every=4,
    )

    # Initialize models
    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    dynamics = Dynamics(**dyn_kwargs)
    task_embedder = TaskEmbedder(d_model=cfg.d_model_dyn, n_agent=cfg.n_agent,
                                 use_ids=cfg.use_task_ids, n_tasks=cfg.n_tasks)
    policy_head_bc = PolicyHeadMTP(d_model=cfg.d_model_dyn, action_dim=cfg.action_dim, L=cfg.L)
    reward_head = RewardHeadMTP(d_model=cfg.d_model_dyn, L=cfg.L, num_bins=cfg.num_reward_bins)
    
    # NEW: RL policy head and value head
    policy_head = PolicyHeadMTP(d_model=cfg.d_model_dyn, action_dim=cfg.action_dim, L=cfg.L)
    value_head = ValueHead(d_model=cfg.d_model_dyn, num_bins=cfg.num_value_bins)

    # Initialize variables for shape inference
    rng = jax.random.PRNGKey(0)
    patches_btnd = temporal_patchify(frames_init, patch)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    fake_z = jnp.zeros((cfg.B, cfg.T, cfg.enc_n_latents, cfg.enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

    # Load tokenizer (encoder/decoder) - get tokenizer_ckpt from BC/rew meta
    # We'll need to read the BC/rew checkpoint meta first to get tokenizer_ckpt path
    bc_rew_mngr = make_manager(cfg.bc_rew_ckpt, item_names=("meta",))
    bc_rew_latest = bc_rew_mngr.latest_step()
    if bc_rew_latest is None:
        raise FileNotFoundError(f"No BC/rew checkpoint found in {cfg.bc_rew_ckpt}")
    bc_rew_meta_restored = bc_rew_mngr.restore(bc_rew_latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    bc_rew_meta = bc_rew_meta_restored.meta
    tokenizer_ckpt = bc_rew_meta.get("tokenizer_ckpt_dir") or bc_rew_meta.get("cfg", {}).get("tokenizer_ckpt")
    if tokenizer_ckpt is None:
        raise ValueError(f"Could not find tokenizer_ckpt in BC/rew checkpoint meta: {bc_rew_meta}")

    enc_vars, dec_vars, tokenizer_meta = load_pretrained_tokenizer(
        tokenizer_ckpt, rng=rng,
        encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    # Build initial z1 to shape dynamics init
    mae_eval_key = jax.random.PRNGKey(777)
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((cfg.B, cfg.T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((cfg.B, cfg.T), k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions_init, step_idx, sigma_idx, z1)
    
    # Initialize task_embedder, BC policy head, reward head
    rng_task, rng_pi_bc, rng_rw = jax.random.split(jax.random.PRNGKey(1), 3)
    dummy_task_ids = jnp.zeros((cfg.B,), dtype=jnp.int32)
    task_vars = task_embedder.init({"params": rng_task}, dummy_task_ids, cfg.B, cfg.T)
    
    fake_h = jnp.zeros((cfg.B, cfg.T, cfg.d_model_dyn), dtype=jnp.float32)
    pi_bc_vars = policy_head_bc.init({"params": rng_pi_bc, "dropout": rng_pi_bc}, fake_h, deterministic=True)
    rew_vars = reward_head.init({"params": rng_rw, "dropout": rng_rw}, fake_h, deterministic=True)

    # Load BC/rew checkpoint (dynamics, task_embedder, BC policy head, reward head)
    dyn_vars, task_vars, pi_bc_vars, rew_vars, bc_rew_meta = load_bc_rew_checkpoint(
        cfg.bc_rew_ckpt, rng=rng,
        dynamics=dynamics, task_embedder=task_embedder,
        policy_head_bc=policy_head_bc, reward_head=reward_head,
        dyn_vars=dyn_vars, task_vars=task_vars,
        pi_bc_vars=pi_bc_vars, rew_vars=rew_vars,
        sample_actions=actions_init, sample_z1=z1,
    )

    # Initialize NEW RL policy head and value head (trainable)
    rng_pi, rng_val = jax.random.split(jax.random.PRNGKey(2), 2)
    pi_vars = policy_head.init({"params": rng_pi, "dropout": rng_pi}, fake_h, deterministic=True)
    val_vars = value_head.init({"params": rng_val, "dropout": rng_val}, fake_h, deterministic=True)

    # Pack params for optimizer (only trainable modules: pi and val)
    params = {
        "pi": pi_vars["params"],
        "val": val_vars["params"],
    }

    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)

    return TrainState(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        task_embedder=task_embedder,
        policy_head_bc=policy_head_bc,
        reward_head=reward_head,
        policy_head=policy_head,
        value_head=value_head,
        enc_vars=enc_vars,
        dec_vars=dec_vars,
        dyn_vars=dyn_vars,
        task_vars=task_vars,
        pi_bc_vars=pi_bc_vars,
        rew_vars=rew_vars,
        pi_vars=pi_vars,
        val_vars=val_vars,
        params=params,
        enc_kwargs=enc_kwargs,
        dec_kwargs=dec_kwargs,
        dyn_kwargs=dyn_kwargs,
        tx=tx,
        opt_state=opt_state,
        mae_eval_key=mae_eval_key,
    )

# ---------------------------
# Meta for RL checkpoints
# ---------------------------

def make_rl_meta(
    *,
    enc_kwargs: dict,
    dec_kwargs: dict,
    dynamics_kwargs: dict,
    H: int, W: int, C: int,
    patch: int,
    k_max: int,
    packing_factor: int,
    n_spatial: int,
    bc_rew_ckpt_dir: str | None = None,
    cfg: Dict[str, Any] | None = None,
):
    return {
        "enc_kwargs": enc_kwargs,
        "dec_kwargs": dec_kwargs,
        "dynamics_kwargs": dynamics_kwargs,
        "H": H, "W": W, "C": C, "patch": patch,
        "k_max": k_max,
        "packing_factor": packing_factor,
        "n_spatial": n_spatial,
        "bc_rew_ckpt_dir": bc_rew_ckpt_dir,
        "cfg": cfg or {},
    }

# ---------------------------
# Main
# ---------------------------

def run(cfg: RLConfig):
    # Initialize wandb if enabled
    if cfg.use_wandb:
        if not WANDB_AVAILABLE:
            print("[warning] wandb requested but not installed. Install with: pip install wandb")
            print("[warning] Continuing without wandb logging.")
        else:
            wandb_project = cfg.wandb_project or cfg.run_name
            wandb.init(
                entity=cfg.wandb_entity,
                project=wandb_project,
                name=cfg.run_name,
                config=asdict(cfg),
                dir=str(Path(cfg.log_dir).resolve()),
            )
            print(f"[wandb] Initialized run: {wandb.run.name if wandb.run else 'N/A'}")

    # Output dirs
    root = _ensure_dir(Path(cfg.log_dir))
    run_dir = _ensure_dir(root / cfg.run_name)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    print(f"[setup] writing artifacts to: {run_dir.resolve()}")

    # Data iterator (streaming)
    next_batch = make_iterator(
        cfg.B, cfg.T, cfg.H, cfg.W, cfg.C,
        pixels_per_step=cfg.pixels_per_step,
        size_min=cfg.size_min, size_max=cfg.size_max,
        hold_min=cfg.hold_min, hold_max=cfg.hold_max,
        fg_min_color=0 if cfg.diversify_data else 128,
        fg_max_color=255 if cfg.diversify_data else 128,
        bg_min_color=0 if cfg.diversify_data else 255,
        bg_max_color=255 if cfg.diversify_data else 255,
    )

    # Initialize models and load checkpoints
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init, rewards_init) = next_batch(init_rng)

    train_state = initialize_models(cfg, frames_init, actions_init)

    # Extract some values for checkpointing
    patch = cfg.patch
    k_max = cfg.k_max
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    # -------- Orbax manager & (optional) restore --------
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)
    meta = make_rl_meta(
        enc_kwargs=train_state.enc_kwargs,
        dec_kwargs=train_state.dec_kwargs,
        dynamics_kwargs=train_state.dyn_kwargs,
        H=cfg.H, W=cfg.W, C=cfg.C, patch=patch,
        k_max=k_max, packing_factor=cfg.packing_factor, n_spatial=n_spatial,
        bc_rew_ckpt_dir=cfg.bc_rew_ckpt,
        cfg=asdict(cfg),
    )

    rng = jax.random.PRNGKey(0)
    state_example = make_state(train_state.params, train_state.opt_state, rng, step=0)
    restored = try_restore(mngr, state_example, meta)

    start_step = 0
    if restored is not None:
        latest_step, r = restored
        train_state.params = r.state["params"]
        train_state.opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"]) + 1
        # Bind subtrees into module vars
        train_state.pi_vars = with_params(train_state.pi_vars, train_state.params["pi"])
        train_state.val_vars = with_params(train_state.val_vars, train_state.params["val"])
        print(f"[restore] Resumed from {ckpt_dir} at step={latest_step}")

    # -------- Training loop --------
    train_rng = jax.random.PRNGKey(2025)
    data_rng = jax.random.PRNGKey(12345)

    start_wall = time.time()
    for step in range(start_step, cfg.max_steps + 1):
        # Sample a batch of videos
        data_rng, batch_key = jax.random.split(data_rng)
        _, (videos, actions_full, rewards_full) = next_batch(batch_key)
        
        # Sample contexts from videos (per-video random start indices)
        data_rng, ctx_key = jax.random.split(data_rng)
        context_frames, context_actions, context_rewards = sample_contexts(
            videos, actions_full, rewards_full, ctx_key, cfg.context_length, cfg.H, cfg.W, cfg.C
        )
        import ipdb; ipdb.set_trace()
        
        # TODO: Implement training step
        # - Encode context_frames into latents using encoder
        # - Generate imagined rollouts using dynamics + policy_head
        # - Compute rewards using reward_head (frozen)
        # - Compute values using value_head (trainable)
        # - Compute TD(lambda) returns
        # - Train policy_head and value_head
        # - Apply behavioral prior regularization (KL divergence from policy_head_bc)
        
        if step % cfg.log_every == 0:
            print(f"[train] step={step:06d} | TODO: Implement training step")
            
            if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"step": step}, step=step)

        # Save checkpoint
        state = make_state(train_state.params, train_state.opt_state, train_rng, step)
        maybe_save(mngr, step, state, meta)

    # Ensure all writes finished
    mngr.wait_until_finished()

    # Save final config
    (run_dir / "config.txt").write_text("\n".join([f"{k}={v}" for k, v in asdict(cfg).items()]))

    # Finish wandb run
    if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("[wandb] Finished logging.")


if __name__ == "__main__":
    cfg = RLConfig(
        run_name="train_policy",
        bc_rew_ckpt="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs/train_bc_rew/checkpoints",
        use_wandb=False,
        wandb_entity="edhu",
        wandb_project="tiny_dreamer_4",
        log_dir="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs",
        max_steps=1_000_000_000,
        log_every=5_000,
        lr=1e-4,
        write_video_every=100_000,
        ckpt_save_every=100_000,
        ckpt_max_to_keep=2,
    )
    print("Running RL config:\n  " + "\n  ".join([f"{k}={v}" for k,v in asdict(cfg).items()]))
    run(cfg)
