# Milestone 1

Before we can start doing any implementation we need to get acquainted with the architecture that we are going to use. In order to have an actual functioning model at the end of this milestone we are going to download the weights from [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). According to the [config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json), the model architecture is LlamaForCausalLM we can inspect that in the [HF repo](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

The Transformer block looks like the following:
```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#ffffff", "primaryTextColor": "#ffffff", "lineColor": "#334155"}}}%%
flowchart TB
    input["Input hidden states"] --> norm1["RMSNorm / LayerNorm"]
    norm1 --> attn["Attention with RoPE"]
    input -. residual .-> add1@{ shape: f-circ, label: "+" }
    attn --> add1
    add1 --> norm2["RMSNorm / LayerNorm"]
    norm2 --> ffn["MLP with SwiGLU"]
    add1 -. residual .-> add2@{ shape: f-circ, label: "+" }
    ffn --> add2
    add2 --> output["Output hidden states"]

    classDef io fill:#1e3a5f,stroke:#0f172a,stroke-width:2px,color:#ffffff;
    classDef norm fill:#115e59,stroke:#042f2e,stroke-width:2px,color:#ffffff;
    classDef attention fill:#4c1d95,stroke:#2e1065,stroke-width:2px,color:#ffffff;
    classDef residual fill:#7c2d12,stroke:#431407,stroke-width:3px,color:#ffffff;
    classDef ffn fill:#86198f,stroke:#4a044e,stroke-width:2px,color:#ffffff;

    class input,output io;
    class norm1,norm2 norm;
    class attn attention;
    class add1,add2 residual;
    class ffn ffn;
```

From this we can start implementing the individual components



