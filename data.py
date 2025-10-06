""" Create a synthetic dataset"""
from einops import rearrange
import jax
import jax.numpy as jnp
from functools import partial
from jax.random import randint, split
import imageio

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

@partial(
    jax.jit,
    static_argnames=["batch_size", "time_steps", "height", "width", "channels", "square_size"]
)
def generate_batch(
    init_pos: jnp.ndarray,               # (B, 2) int32, top-left (y,x)
    init_vel: jnp.ndarray,               # (B, 2) int32, (vy, vx)
    init_background_color: jnp.ndarray,  # (B, C) uint8
    init_foreground_color: jnp.ndarray,  # (B, C) uint8
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
    channels: int,
    square_size: int,                    # k (side length of the square)
) -> jnp.ndarray:
    """
    Generates a (B, T, H, W, C) video where a k×k square bounces inside the frame.
    Positions are for the *top-left* corner of the square.
    """

    H, W, k = height, width, square_size
    # Valid top-left positions after reflection:
    # y in [0, H - k], x in [0, W - k]
    max_y = H - k
    max_x = W - k

    # integrate the position and velocity over time, for one sample
    def step(carry, _):
        pos, vel = carry            # pos=(y,x), vel=(vy,vx)
        y, x = pos
        vy, vx = vel

        # Propose next position
        y_next = y + vy
        x_next = x + vx

        # --- reflect Y against 0 and max_y ---
        # below 0: mirror to -y_next, flip vy
        vy = jnp.where(y_next < 0, -vy, vy)
        y_next = jnp.where(y_next < 0, -y_next, y_next)
        # above max_y: mirror around max_y, flip vy
        vy = jnp.where(y_next > max_y, -vy, vy)
        y_next = jnp.where(y_next > max_y, 2 * max_y - y_next, y_next)

        # --- reflect X against 0 and max_x ---
        vx = jnp.where(x_next < 0, -vx, vx)
        x_next = jnp.where(x_next < 0, -x_next, x_next)
        vx = jnp.where(x_next > max_x, -vx, vx)
        x_next = jnp.where(x_next > max_x, 2 * max_x - x_next, x_next)

        pos_next = jnp.stack([y_next, x_next])
        vel_next = jnp.stack([vy, vx])
        return (pos_next, vel_next), pos_next

    def integrate_positions(p, v):
        (_, _), positions = jax.lax.scan(step, (p, v), jnp.arange(time_steps))
        return positions  # (T, 2)

    # (B, T, 2)
    positions = jax.vmap(integrate_positions)(init_pos, init_vel)

    # Initialize background
    video = (
        jnp.ones((batch_size, time_steps, H, W, channels), dtype=jnp.uint8)
        * init_background_color[:, None, None, None, :]
    )

    # Painter for one frame: write a k×k stamp at (y, x)
    def paint_one_frame(frame, y, x, color):
        # stamp shape (k, k, C)
        stamp = jnp.broadcast_to(color, (k, k, channels))
        return lax.dynamic_update_slice(frame, stamp, (y, x, 0))

    # Vectorize over time, then batch. Colors are per-batch.
    paint_over_time = jax.vmap(
        paint_one_frame,
        in_axes=(0,    0,    0,    None),  # frame_t, y_t, x_t, color_b
        out_axes=0
    )
    paint_over_batch = jax.vmap(
        paint_over_time,
        in_axes=(0,    0,       0,       0),  # video_b, y_bt, x_bt, color_b
        out_axes=0
    )

    # Split positions
    y_idx = positions[..., 0]  # (B, T)
    x_idx = positions[..., 1]  # (B, T)

    # Apply painting
    video = paint_over_batch(video, y_idx, x_idx, init_foreground_color)

    # Normalize to [0,1]
    return video.astype(jnp.float32) / 255.0


# @partial(jax.jit, static_argnames=["batch_size", "time_steps", "height", "width", "channels"])
# def generate_batch(init_pos: jnp.ndarray, init_vel: jnp.ndarray, init_background_color: jnp.ndarray, init_foreground_color: jnp.ndarray, batch_size: int, time_steps: int, height: int, width: int, channels: int) -> jnp.ndarray:
#     # integrate the position and velocity over time, for one sample
#     def step(carry, _):
#         pos, vel = carry            # pos=(y,x), vel=(vy,vx)  (ints)
#         y, x = pos
#         vy, vx = vel

#         # Propose next position
#         y_next = y + vy
#         x_next = x + vx

#         # --- reflect Y (horizontal walls at 0 and H-1) ---
#         # If we overshoot below 0: mirror position to -y_next, flip vy
#         vy = jnp.where(y_next < 0, -vy, vy)
#         y_next = jnp.where(y_next < 0, -y_next, y_next)
#         # If we overshoot above H-1: mirror around H-1, flip vy
#         vy = jnp.where(y_next >= height, -vy, vy)
#         y_next = jnp.where(y_next >= height, 2 * (height - 1) - y_next, y_next)

#         # --- reflect X (vertical walls at 0 and W-1) ---
#         vx = jnp.where(x_next < 0, -vx, vx)
#         x_next = jnp.where(x_next < 0, -x_next, x_next)
#         vx = jnp.where(x_next >= width, -vx, vx)
#         x_next = jnp.where(x_next >= width, 2 * (width - 1) - x_next, x_next)

#         pos_next = jnp.stack([y_next, x_next])
#         vel_next = jnp.stack([vy, vx])
#         return (pos_next, vel_next), pos_next
    
#     def integrate_and_render(p, v):
#         return jax.lax.scan(step, (p, v), jnp.arange(time_steps))
    
#     batched_integrate_and_render = jax.vmap(integrate_and_render)
#     _, positions = batched_integrate_and_render(init_pos, init_vel)
#     # first set the background color
#     video = jnp.ones((batch_size, time_steps, height, width, channels), dtype=jnp.uint8) * init_background_color[:, None, None, None]
#     # do a batched scatter
#     b_idx = jnp.arange(batch_size)[:, None]          # (B, 1)
#     t_idx = jnp.arange(time_steps)[None, :]          # (1, T)
#     y_idx = positions[..., 0] # (B, T)
#     x_idx = positions[..., 1] # (B, T)
#     video = video.at[b_idx, t_idx, y_idx, x_idx, :].set(init_foreground_color[:, None])
#     # normalize video to [0, 1 ] for pixel values
#     video = video / 255.0
#     return video

def make_iterator(batch_size, time_steps, height, width, channels, square_size):
    # Optionally, partially apply your jitted generate_batch so sizes are baked in
    gen = jax.jit(
        lambda pos, vel, bg, fg: generate_batch(
            pos, vel, bg, fg, batch_size, time_steps, height, width, channels, square_size
        )
    )

    @jax.jit
    def next(key):
        key, sub = jax.random.split(key)
        k_pos, k_vel, k_bg, k_fg = jax.random.split(sub, 4)

        init_pos = jax.random.randint(
            k_pos, (batch_size, 2),
            minval=jnp.array([0, 0]), maxval=jnp.array([height, width])
        )
        init_vel = jax.random.randint(
            k_vel, (batch_size, 2),
            minval=jnp.array([-3, -3]), maxval=jnp.array([4, 4])
        )
        init_bg = jax.random.randint(
            k_bg, (batch_size, 3),
            minval=jnp.array([0, 0, 0]), maxval=jnp.array([255, 255, 255]),
            dtype=jnp.uint8
        )
        init_fg = jax.random.randint(
            k_fg, (batch_size, 3),
            minval=jnp.array([0, 0, 0]), maxval=jnp.array([255, 255, 255]),
            dtype=jnp.uint8
        )

        video = gen(init_pos, init_vel, init_bg, init_fg)
        return key, video

    return next
 

def patchify(x: jnp.ndarray, patch: int) -> jnp.ndarray:
    """
    x: (B, H, W, C)  ->  patches: (B, N, D)
      where N = (H/patch)*(W/patch), D = patch*patch*C
    """
    patches = rearrange(x, "b (hp p1) (wp p2) c -> b (hp wp) (p1 p2 c)", p1=patch, p2=patch)
    return patches

def unpatchify(patches: jnp.ndarray, H: int, W: int, C: int, patch: int) -> jnp.ndarray:
    """
    patches: (B, N, D)  ->  x: (B, H, W, C)
      where N = (H/patch)*(W/patch), D = patch*patch*C
    """
    image = rearrange(patches, "b (hp wp) (p1 p2 c) -> b (hp p1) (wp p2) c", hp=H//patch, wp=W//patch, p1=patch, p2=patch, c=C)
    return image


def test_generate_batch():
    key = jax.random.PRNGKey(0)
    batch_size = 8
    time_steps = 64
    height = 16
    width = 16
    channels = 3
    # randomly choose an initial position and velocity
    pos_key, vel_key, background_key, foreground_key = split(key, 4)
    init_pos = randint(pos_key, shape=(batch_size, 2), minval=jnp.array([0, 0]), maxval=jnp.array([height, width]))
    init_vel = randint(vel_key, shape=(batch_size, 2), minval=jnp.array([-3, -3]), maxval=jnp.array([3 + 1, 3 + 1]))
    init_background_color = randint(background_key, shape=(batch_size, 3), minval=jnp.array([0, 0, 0]), maxval=jnp.array([255, 255, 255]), dtype=jnp.uint8)
    init_foreground_color = randint(foreground_key, shape=(batch_size, 3), minval=jnp.array([0, 0, 0]), maxval=jnp.array([255, 255, 255]), dtype=jnp.uint8)
    video = generate_batch(init_pos, init_vel, init_background_color, init_foreground_color, batch_size, time_steps, height, width, channels, 4)
    # render the batch of video (B, T, H, W, C) as a row of images per timestep.
    def render_frame(_, frame):
        grid = jnp.concatenate(frame, axis=1)
        return (), grid
    _, all_imgs = jax.lax.scan(render_frame, (), video.transpose(1, 0, 2, 3, 4))
    
    # save the video as a gif
    imageio.mimsave('video.gif', jnp.asarray(all_imgs * 255.0, dtype=jnp.uint8), fps=8)

def test_patchify():
    B = 2
    H = 16
    W = 16
    C = 3
    patch = 8
    assert H % patch == 0 and W % patch == 0, "H,W must be multiples of patch"
    x = jnp.ones((B, H, W, C), dtype=jnp.uint8)
    print(f"x.shape: {x.shape}")
    patches = patchify(x, patch)
    hp = H // patch
    wp = W // patch
    D = patch * patch * C
    assert patches.shape == (B, hp * wp, D), f"expected ({B}, {hp * wp}, {D}), got {patches.shape}"
    print(f"patches.shape: {patches.shape}")
    recovered_x = unpatchify(patches, H, W, C, patch)
    assert x.shape == (B,H,W,C), f"expected ({B}, {H}, {W}, {C}), got {recovered_x.shape}"
    print(f"x.shape: {x.shape}")


if __name__ == "__main__":
    test_generate_batch()
    # test_patchify()