import jax 
import jax.numpy as jnp
import numpy as np

from module import Module, Linear
from encoding import apply_rope

class SiLU(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: jnp.ndarray):

        sigmoid = 1.0 / (1.0 + jnp.exp(-x))
        return x * sigmoid

class FFNSwiGLU(Module):
    def __init__(self, params: np.ndarray):
        super().__init__()

        self.act_fn = SiLU()
        self.up_proj = Linear(params)
        self.gate_proj = Linear(params)
        self.down_proj = Linear(params)

    def __call__(self, x: jnp.ndarray):

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

    def __call__(self, x: jnp.ndarray, rope_matrix: jnp.ndarray, attention_mask: jnp.ndarray):

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
    def __init__(self, eps: float = 1e-8, hidden_size: int = 512):
        super().__init__()

        self.eps = eps
        self.weight = jnp.ones((hidden_size))

    def __call__(self, x: jnp.ndarray):

        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x_normalized = x / jnp.sqrt(variance + self.eps)

        return x_normalized * self.weight