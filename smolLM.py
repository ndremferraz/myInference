import jax.numpy as jnp
import numpy as np

import jax.dlpack as jdl
import torch

from module import Module, Embedding
from llama import RotaryEmbedding, LLamaTransformer, LlamaRMSNorm
from safetensors import safe_open

def causal_mask(seq_len):
    mask = jnp.triu(
        jnp.ones((seq_len, seq_len), dtype=bool),
        k=1
    )

    return jnp.where(mask, -jnp.inf, 0.0)

def torch_to_jax(tensor: torch.Tensor):

    return jdl.from_dlpack(tensor)

def init_transformer_layer(f, i, rms_norm_eps, device):

    input_ln_name = f'model.layers.{i}.input_layernorm.weight'
    
    mlp_down_name = f'model.layers.{i}.mlp.down_proj.weight'
    mlp_gate_name = f'model.layers.{i}.mlp.gate_proj.weight'
    mlp_up_name = f'model.layers.{i}.mlp.up_proj.weight'

    post_attn_ln_name = f'model.layers.{i}.post_attention_layernorm.weight'

    k_proj_name = f'model.layers.{i}.self_attn.k_proj.weight'
    o_proj_name = f'model.layers.{i}.self_attn.o_proj.weight'
    q_proj_name = f'model.layers.{i}.self_attn.q_proj.weight'
    v_proj_name = f'model.layers.{i}.self_attn.v_proj.weight'

    input_ln = f.get_tensor(input_ln_name).to(device)

    mlp_down = f.get_tensor(mlp_down_name).to(device)
    mlp_gate = f.get_tensor(mlp_gate_name).to(device)
    mlp_up = f.get_tensor(mlp_up_name).to(device)

    post_attn_ln = f.get_tensor(post_attn_ln_name).to(device)

    k_proj = f.get_tensor(k_proj_name).to(device)
    o_proj = f.get_tensor(o_proj_name).to(device)
    q_proj = f.get_tensor(q_proj_name).to(device)
    v_proj = f.get_tensor(v_proj_name).to(device)

    return LLamaTransformer(wq=torch_to_jax(q_proj),
                            wk=torch_to_jax(k_proj),
                            wv=torch_to_jax(v_proj),
                            wo=torch_to_jax(o_proj),
                            w_up=torch_to_jax(mlp_up),
                            w_gate=torch_to_jax(mlp_gate),
                            w_down=torch_to_jax(mlp_down),
                            input_ln_weights=torch_to_jax(input_ln),
                            post_attn_ln_weights=torch_to_jax(post_attn_ln),
                            rms_norm_eps=rms_norm_eps)

class SmolLM(Module):

    def __init__(self, 
                 safetensor_filepath: str,
                 num_hidden_layers: int = 8,
                 rope_theta: float = 10000.0,
                 rms_norm_eps: float = 1e-8,
                 hidden_size: int = 512,
                 device: str = "cpu"):
        super().__init__()

        self.device = device

        with safe_open(safetensor_filepath, framework="pt") as f:

            tensors = f.keys()

            embedding_weights = f.get_tensor(tensors.pop(0)).to(self.device)

            self.embedding = Embedding(torch_to_jax(embedding_weights))
            self.rope_encoder = RotaryEmbedding(dims=hidden_size, rope_theta=rope_theta)

            layer_list = []

            for i in range(num_hidden_layers):

                layer = init_transformer_layer(f, i, rms_norm_eps, self.device)
                layer_list.append(layer)

            self.layers = layer_list

            norm_weights = f.get_tensor(tensors.pop()).to(self.device)
            self.last_ln = LlamaRMSNorm(eps=rms_norm_eps, norm_weights=torch_to_jax(norm_weights))

    def __call__(self, x):

        position_ids = jnp.arange(x.shape[1])
        attn_mask = causal_mask(x.shape[1])

        x = self.embedding(x)
        rope_matrix = self.rope_encoder(x=x, position_ids=position_ids)

        for layer in self.layers:
            x = layer(x, rope_matrix, attn_mask)

        x = self.last_ln(x)

        return x

