# Milestone 1

Before we can start doing any implementation we need to get acquainted with the architecture that we are going to use. In order to have an actual functioning model at the end of this milestone we are going to download the weights from [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). According to the [config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json), the model architecture is LlamaForCausalLM, the source code can be found [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

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

Some of the components like RoPE, RMSNorm, and SwiGLU gates are non-trivial even for those with a basic understanding of LLMs so we are going explain them here. 

## [RoPE(Rotary Positional Embedding)](https://arxiv.org/pdf/1910.07467)

The Original Attention Mechanism from *Attention is All You Need* ignores the relative position of tokens almost by design, and simply adds absolute positional encoding through context. Let's assume that we are reading a piece of text, as we progress from the 5th to 6th paragraph the words that we read in the 1st paragraph should carry less and less importance. RoPE proposes an alternative Attention that strongly addresses this issue. 

Note: This is not something exclusive to RoPE, but after the *Attention is All You Need* incorporating positional encoding to the transformer blocks rather than exclusively at the bottom of the Encoder/Decoder Stack became a trend. 









