from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class TinyEnvConfig:
    env_type: str = "TinyEnv"
    B: int = 32
    T: int = 64
    H: int = 32
    W: int = 32
    C: int = 3
    pixels_per_step: int = 2  # how many pixels the agent moves per step
    size_min: int = 6         # minimum size of the square
    size_max: int = 14        # maximum size of the square
    hold_min: int = 4         # how long the agent holds a direction for
    hold_max: int = 9         # how long the agent holds a direction for
    diversify_data: bool = True
    action_dim: int = 4       # number of categorical actions


@dataclass(frozen=True)
class AtariConfig:
    env_type: str = "Atari"
    env_name: str = "PongNoFrameskip-v4"
    B: int = 32
    T: int = 32
    H: int = 64
    W: int = 64
    C: int = 3
    frame_stack: int = 4
    grayscale: bool = True
    resize_height: int = 84
    resize_width: int = 84

EnvConfig = Union[TinyEnvConfig, AtariConfig]