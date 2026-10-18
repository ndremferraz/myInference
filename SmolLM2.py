import jax
import jax.numpy as jnp
from flax import nnx

from safetensors import safe_open
import jax.dlpack   

from llama import LLamaTransformer, RotaryPositionalEmbedding, LlamaRMSNorm

class SmolLM2(nnx.Module):


    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 hidden_dim: int,
                 intermediate_dim: int,
                 head_dim: int,
                 q_heads: int, 
                 kv_heads: int, 
                 rms_eps: float = 1e-5, 
                 rope_theta: float = 100000.0):

        self.num_layers = num_layers

        self.embed = nnx.Embed(vocab_size, hidden_dim, rngs=nnx.Rngs(0))
        self.RoPE = RotaryPositionalEmbedding(head_dim, rope_theta)

        self.layers = nnx.List([
            LLamaTransformer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                q_heads=q_heads,
                kv_heads=kv_heads,
                rms_eps=rms_eps,
            )
            for _ in range(num_layers)
        ])

        self.norm = LlamaRMSNorm(hidden_dim, rms_eps)

    def __call__(self, 
                 x: jnp.ndarray, 
                 position_ids: jnp.ndarray, 
                 attention_mask: jnp.ndarray):

        x = self.embed(x)
        rope_matrix = self.RoPE(x, position_ids)

        for layer in self.layers:
            x = layer(x, rope_matrix, attention_mask)

        x = self.norm(x)
        x = self.embed.attend(x)
        
        return x

    def load_from_safetensor(self, safetensor_path: str):

        with safe_open(safetensor_path, framework="pt") as f:

            embed_weight_pt = f.get_tensor("model.embed_tokens.weight")
            embed_weight_jax = jax.dlpack.from_dlpack(embed_weight_pt)
            self.embed.embedding.set_value(embed_weight_jax)

            for i, layer in enumerate(self.layers):

                input_ln_weight_pt = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                mlp_up_proj_weight_pt = f.get_tensor(f"model.layers.{i}.mlp.up_proj.weight")
                mlp_gate_proj_weight_pt = f.get_tensor(f"model.layers.{i}.mlp.gate_proj.weight")
                mlp_down_proj_weight_pt = f.get_tensor(f"model.layers.{i}.mlp.down_proj.weight")
                post_attn_ln_weight_pt = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")

                q_proj_weight_pt = f.get_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
                k_proj_weight_pt = f.get_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
                v_proj_weight_pt = f.get_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
                o_proj_weight_pt = f.get_tensor(f"model.layers.{i}.self_attn.o_proj.weight")

                input_ln_weight_jax = jax.dlpack.from_dlpack(input_ln_weight_pt)
                mlp_up_proj_weight_jax = jax.dlpack.from_dlpack(mlp_up_proj_weight_pt)
                mlp_gate_proj_weight_jax = jax.dlpack.from_dlpack(mlp_gate_proj_weight_pt)
                mlp_down_proj_weight_jax = jax.dlpack.from_dlpack(mlp_down_proj_weight_pt)
                post_attn_ln_weight_jax = jax.dlpack.from_dlpack(post_attn_ln_weight_pt)

                q_proj_weight_jax = jax.dlpack.from_dlpack(q_proj_weight_pt)
                k_proj_weight_jax = jax.dlpack.from_dlpack(k_proj_weight_pt)
                v_proj_weight_jax = jax.dlpack.from_dlpack(v_proj_weight_pt)
                o_proj_weight_jax = jax.dlpack.from_dlpack(o_proj_weight_pt)


                layer.input_lnorm.weight.set_value(input_ln_weight_jax)
                layer.ffn.up_proj.kernel.set_value(mlp_up_proj_weight_jax.T)
                layer.ffn.gate_proj.kernel.set_value(mlp_gate_proj_weight_jax.T)
                layer.ffn.down_proj.kernel.set_value(mlp_down_proj_weight_jax.T)
                layer.post_attn_lnorm.weight.set_value(post_attn_ln_weight_jax)

                layer.attention.q_proj.kernel.set_value(q_proj_weight_jax.T)
                layer.attention.k_proj.kernel.set_value(k_proj_weight_jax.T)
                layer.attention.v_proj.kernel.set_value(v_proj_weight_jax.T)
                layer.attention.o_proj.kernel.set_value(o_proj_weight_jax.T)

            norm_weight_pt = f.get_tensor("model.norm.weight")
            norm_weight_jax = jax.dlpack.from_dlpack(norm_weight_pt)
            self.norm.weight.set_value(norm_weight_jax)

