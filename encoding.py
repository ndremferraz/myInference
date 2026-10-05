import jax 
import jax.numpy as jnp
import numpy as np

from module import Module

def rotate_half(x: jnp.ndarray):

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rope(     q: jnp.ndarray, 
                    k: jnp.ndarray, 
                    cos: jnp.ndarray, 
                    sin: jnp.ndarray,
                    ):

    q_half_neg = rotate_half(q)
    k_half_neg = rotate_half(k)

    q = (q * cos - q_half_neg * sin)
    k = (k * cos + k_half_neg * sin)

    return q, k

class RotaryEmbedding(Module):
    def __init__(   self, 
                    dims: int, 
                    rope_theta: float):
        
        super().__init__()

        self.dims = dims
        self.rope_theta = rope_theta
        self.frequencies = self._compute_frequencies(dims, rope_theta)

    @staticmethod
    def _compute_frequencies(dims: int, rope_theta: float):
            
        return rope_theta ** ( (-1) * jnp.arange(0, dims, 2) )

    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray):

        position_ids_expanded = jnp.expand_dims(position_ids, axis=-1)
        freqs = jnp.repeat(self.frequencies, repeats=2, axis=-1)

        rotation_angles = position_ids_expanded * freqs

        return jnp.cos(rotation_angles), jnp.sin(rotation_angles)
