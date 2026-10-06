import jax 
import jax.numpy as jnp
import numpy as np

from module import Module, Linear

def rotate_half(    x: jnp.ndarray
                    ):

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
                    rope_theta: float
                    ):
        
        super().__init__()

        self.dims = dims
        self.rope_theta = rope_theta
        self.frequencies = self._compute_frequencies(dims, rope_theta)

    @staticmethod
    def _compute_frequencies(
                    dims: int, 
                    rope_theta: float
                    ):
            
        return rope_theta ** ( (-1) * jnp.arange(0, dims, 2) )

    def __call__(   self, 
                    x: jnp.ndarray, 
                    position_ids: jnp.ndarray
                    ):

        position_ids_expanded = jnp.expand_dims(position_ids, axis=-1)
        freqs = jnp.repeat(self.frequencies, repeats=2, axis=-1)

        rotation_angles = position_ids_expanded * freqs

        return jnp.cos(rotation_angles), jnp.sin(rotation_angles)


class SiLU(Module):
    def __init__(self):
        super().__init__()

    def __call__(   self, 
                    x: jnp.ndarray):

        sigmoid = 1.0 / (1.0 + jnp.exp(-x))
        return x * sigmoid


class FFNSwiGLU(Module):
    def __init__(   self,
                    params: np.ndarray):
        
        super().__init__()

        self.act_fn = SiLU()
        self.up_proj = Linear(params)
        self.gate_proj = Linear(params)
        self.down_proj = Linear(params)

    def __call__(   self, 
                    x: jnp.ndarray):

        up = self.up_proj(x)
        gate = self.gate_proj(x)
        gate_act = self.act_fn(gate)
        
        swiglu = up * gate_act

        return self.down_proj(swiglu)


class AttentionRoPE(Module):
    def __init__(   self, 
                    dims: int, 
                    wq = np.ndarray,
                    wk = np.ndarray,
                    wv = np.ndarray,
                    wo = np.ndarray,
                 ):
        super().__init__()

        self.dims = dims

        self.q_proj = Linear(weights=wq)
        self.k_proj = Linear(weights=wk)
        self.v_proj = Linear(weights=wv)
        self.o_proj = Linear(weights=wo)

    def __call__(   self, 
                    x: jnp.ndarray,
                    rope_matrix: jnp.ndarray, 
                    attention_mask: jnp.ndarray):

        cos, sin = rope_matrix

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query, key = apply_rope(query, key, cos, sin)

        attn = jnp.dot(query, key.T) / jnp.sqrt(self.dims)
        attn = attn + attention_mask

        attn_logits = jax.nn.softmax(attn)
        attn_o = self.o_proj(attn_logits @ value)

        return attn_o


class LlamaRMSNorm(Module):
    def __init__(   self, 
                    eps: float = 1e-8, 
                    hidden_size: int = 512):
        
        super().__init__()

        self.eps = eps
        self.weight = jnp.ones((hidden_size))

    def __call__(self, x: jnp.ndarray):

        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x_normalized = x / jnp.sqrt(variance + self.eps)

        return x_normalized * self.weight