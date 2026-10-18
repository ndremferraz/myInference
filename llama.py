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

    cos = cos[..., None, :]
    sin = sin[..., None, :]

    q_half_neg = rotate_half(q)
    k_half_neg = rotate_half(k)

    q = (q * cos + q_half_neg * sin)
    k = (k * cos + k_half_neg * sin)

    return q, k


class RotaryPositionalEmbedding(nnx.Module):

    def __init__(self, dims: int, rope_theta: float):

        self.dims = dims
        self.rope_theta = rope_theta
        self.frequencies = self._compute_frequencies(dims, rope_theta)

    @staticmethod
    def _compute_frequencies(dims: int, rope_theta: float):
            
        return rope_theta ** ( (-1) * jnp.arange(0, dims, 2) / dims)

    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray):

        position_ids_expanded = jnp.expand_dims(position_ids, axis=-1)
        freqs = jnp.concatenate([self.frequencies, self.frequencies])
       
        rotation_angles = position_ids_expanded * freqs
       
        return jnp.cos(rotation_angles), jnp.sin(rotation_angles)


class SiLU(nnx.Module):

    def __call__(self, x: jnp.ndarray):
        return x * jax.nn.sigmoid(x)


class FFNSwiGLU(nnx.Module):

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=False, rngs=nnx.Rngs(0))
        self.gate_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=False, rngs=nnx.Rngs(0))

        self.act_fn = SiLU()
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, use_bias=False, rngs=nnx.Rngs(0))

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

        self.q_proj = nnx.Linear(hidden_dim, q_heads * head_dim, use_bias=False, rngs=nnx.Rngs(0))
        self.k_proj = nnx.Linear(hidden_dim, kv_heads * head_dim, use_bias=False, rngs=nnx.Rngs(0))
        self.v_proj = nnx.Linear(hidden_dim, kv_heads * head_dim, use_bias=False, rngs=nnx.Rngs(0))
        self.o_proj = nnx.Linear(q_heads * head_dim, hidden_dim, use_bias=False, rngs=nnx.Rngs(0))

    def __call__(self,
                 x: jnp.ndarray, 
                 rope_matrix: jnp.ndarray,
                 attention_mask: jnp.ndarray):

        cos, sin = rope_matrix

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(*q.shape[:-1], self.q_heads, self.head_dim)
        k = k.reshape(*k.shape[:-1], self.kv_heads, self.head_dim)
        v = v.reshape(*v.shape[:-1], self.kv_heads, self.head_dim)

        q, k = apply_rope(q, k, cos, sin)

        k = jnp.repeat(k, self.q_heads // self.kv_heads, axis=-2)
        v = jnp.repeat(v, self.q_heads // self.kv_heads, axis=-2)

        attn = jnp.einsum('...thd,...shd->...hts', q, k) / jnp.sqrt(self.head_dim)
        attn = attn + attention_mask

        attn_logits = jax.nn.softmax(attn, axis=-1)

        attn_o = jnp.einsum('...hts,...shd->...thd', attn_logits, v)

        attn_o = attn_o.reshape(*attn_o.shape[:-2], self.q_heads * self.head_dim)

        return self.o_proj(attn_o)


class LlamaRMSNorm(nnx.Module):

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(hidden_dim))

    def __call__(self, x: jnp.ndarray):

        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return self.weight * x


class LLamaTransformer(nnx.Module):

    def __init__(self, 
                 hidden_dim: int,
                 intermediate_dim: int,
                 head_dim: int,
                 q_heads: int,
                 kv_heads: int,
                 rms_eps: float = 1e-6,
                 ):

        self.attention = RoPEAttention(hidden_dim, head_dim, q_heads, kv_heads)
        self.ffn = FFNSwiGLU(hidden_dim, intermediate_dim)

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
