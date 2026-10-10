import jax.numpy as jnp
import jax
from flax import nnx

def rotate_half(x):

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


class RotaryPositionalEmbedding(nnx.Module):

    def __init__(self, dims: int, rope_theta: float):

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


class SiLU(nnx.Module):

    def __call__(self, x: jnp.ndarray):
        return x * jax.nn.sigmoid(x)


class FFNSwiGLU(nnx.Module):

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim

        self.up_proj = nnx.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nnx.Linear(hidden_dim, hidden_dim)

        self.act_fn = SiLU()
        self.down_proj = nnx.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: jnp.ndarray):
        x1 = self.up_proj(x)
        x2 = self.gate_proj(x)
        x3 = self.down_proj(x1 * self.act_fn(x2))
        return x3

class RoPEAttention(nnx.Module):
    def __init__(self,
                 hidden_dim: int,
                 head_dim: int,
                 q_heads: int,
                 kv_heads: int):

        self.q_heads = q_heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nnx.Linear(hidden_dim, q_heads * head_dim)
        self.k_proj = nnx.Linear(hidden_dim, kv_heads * head_dim)
        self.v_proj = nnx.Linear(hidden_dim, kv_heads * head_dim)
        self.o_proj = nnx.Linear(q_heads * head_dim, hidden_dim)

    def __call__(self,
                 x: jnp.ndarray, 
                 rope_matrix: jnp.ndarray,
                 attention_mask: jnp.ndarray):

        cos, sin = rope_matrix

        '''
        I STILL NEED TO IMPLEMENT GQA AND SEE HOW IT PAIRS WITH ROPE
        '''

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(*q.shape[:-1], self.q_heads // self.kv_heads, self.kv_heads * self.head_dim)

        attn = q @ k.mT / jnp.sqrt(self.head_dim * self.kv_heads)
        attn = attn + attention_mask

        attn_logprobs = jax.nn.log_softmax(attn, axis=-1)

        attn_o = attn_logprobs @ v

        return attn_o


class LlamaRMSNorm(nnx.Module):

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nnx.Parameter(jnp.ones(hidden_dim))

    def __call__(self, x: jnp.ndarray):

        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return self.weight * x


class LLamaTransformer(nnx.Module):

    def __init__(self, 
                 hidden_dim: int,
                 head_dim: int,
                 q_heads: int,
                 kv_heads: int,
                 rms_eps: float = 1e-6,
                 rope_theta: float = 10000.0,
                 ):

        self.attention = RoPEAttention(hidden_dim, head_dim, q_heads, kv_heads)
        self.ffn = FFNSwiGLU(hidden_dim)

        self.input_lnorm = LlamaRMSNorm(hidden_dim, rms_eps)
        self.post_attn_lnorm = LlamaRMSNorm(hidden_dim, rms_eps)

    def __call__(self, x: jnp.ndarray, rope_matrix: jnp.ndarray, attention_mask: jnp.ndarray):

        residual = x
        x = self.input_lnorm(x)

        x = self.attention(x, rope_matrix, attention_mask)
        x = x + residual
        residual = x

        x = self.post_attn_lnorm(x)
        x = self.ffn(x)

        x = x + residual
        return x
