from dataclasses import dataclass
from typing import List, Optional, Union
from configs.env import *

@dataclass(frozen=True)
class TokenizerConfig:
    run_name: str = "tokenizer_tinyenv"
    log_dir: str = "/storage/inhokim/dreamer4-tinyenv/tokenizer"
    ckpt_max_to_keep: int = 5
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None

    # environment config
    env: EnvConfig = TinyEnvConfig()

    # tokenizer model config
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model: int = 64
    n_heads: int = 4
    enc_depth: int = 8
    dec_depth: int = 8
    enc_dropout: float = 0.05
    dec_dropout: float = 0.05

    # train
    seed: int = 0
    max_steps: int = 500_000
    log_every: int = 100
    viz_every: int = 50_000
    lr: float = 1e-4

    # losses
    lpips_weight: float = 0.2
    lpips_batch_size: int = 8


@dataclass(frozen=True)
class MTPConfig:
    L: int = 2                      # predict next L actions/rewards
    num_reward_bins: int = 101      # twohot bins for symexp rewards
    reward_log_low: float = -3.0    # log-space lower bound for reward bins (tune per dataset)
    reward_log_high: float = 3.0    # log-space upper bound for reward bins (tune per dataset)
    num_value_bins: int = 101       # twohot bins for symexp values
    n_tasks: int = 128              # task-ID space for TaskEmbedder
    use_task_ids: bool = True       # True: discrete task IDs; False: vector embed


@dataclass(frozen=True)
class RealismConfig:
    # IO / ckpt
    run_name: str = "dynamics_tinyenv"
    tokenizer_ckpt: str = "/storage/inhokim/dreamer4-tinyenv/<tokenizer_path>"
    pretrained_dyn_ckpt: str = "/storage/inhokim/dreamer4-tinyenv/<dynamics_path>"
    log_dir: str = "./<log_path>"
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 50_000


    # wandb config
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None

    # environment config
    env: EnvConfig = TinyEnvConfig()

    # tokenizer config
    tokenizer: TokenizerConfig = TokenizerConfig()

    # dynamics config
    d_model: int = 128
    depth: int = 8
    packing_factor: int = 2
    n_register: int = 4 # number of register tokens for dynamics
    n_agent: int = 1 # number of agent tokens for dynamics

    # UPDATED: default to wm_agent (fine-tuning with agent readouts)
    agent_space_mode: str = "wm_agent"

    # schedule
    k_max: int = 8
    bootstrap_start: int = 5_000  # warm-up steps with bootstrap masked out
    self_fraction: float = 0.25   # used once we pass bootstrap_start

    # train
    seed: int = 0
    max_steps: int = 1_000_000
    log_every: int = 5_000
    lr: float = 1e-4

    # eval media toggle
    write_video_every: int = 50_000  # set large to reduce IO, or 0 to disable entirely

    # NEW: multi-token prediction (MTP) settings
    mtp: MTPConfig = MTPConfig()

    # Loss weighting (to balance scales across different loss components)
    loss_weight_shortcut: float = 1.0    # weight for flow/bootstrap loss (MSE units)
    loss_weight_policy: float = 0.3      # weight for policy CE loss (nats)
    loss_weight_reward: float = 0.3      # weight for reward CE loss (nats)


@dataclass(frozen=True)
class RLConfig:
    # IO / ckpt
    run_name: str
    bc_rew_ckpt: str  # checkpoint from train_bc_rew_heads.py
    log_dir: str = "./logs"
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None

    # environment config
    env: EnvConfig = TinyEnvConfig()

    # tokenizer / dynamics config
    dynamics: RealismConfig = RealismConfig()
    tokenizer: TokenizerConfig = dynamics.tokenizer

    # schedule
    k_max: int = 8

    # train
    max_steps: int = 1_000_000_000
    log_every: int = 5_000
    lr: float = 3e-4

    # eval media toggle
    write_video_every: int = 10_000
    visualize_every: int = 25_000

    # RL-specific
    mtp: MTPConfig = MTPConfig()

    # RL hyperparameters
    gamma: float = 0.997
    lambda_: float = 0.95
    horizon: int = 32
    context_length: int = 16
    imagination_d: float = 1.0 / 4
    alpha: float = 0.5
    beta: float = 0.3

    # Evaluation
    eval_every: int = 50_000
    eval_episodes: int = 4
    eval_horizon: int = 32
    eval_batch_size: int = 4
    max_eval_examples_to_plot: int = 4
