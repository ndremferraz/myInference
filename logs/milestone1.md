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

RoPE is a fairly complex concept, so I recommend watching this [Video](https://www.youtube.com/watch?v=GQPOtyITy54). The underlying principle is to rotate(via rotation matrix) the embedding vectors by an angle that is proportional to their position. The intuition is the vectors with a greater angle between them have a lesser dot product. 

The way that this gets done is by grouping every two dimensions in the embeddings together and rotating them just like one would rotate a two dimensional vector. For instance, the dimensions $$j$$ and $$j+1$$ of a query embedding are rotated is the following:

$$
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
q_j \\
q_{j+1}
\end{bmatrix}
$$

where $$m$$ is the respective position of the token in the sequence and $i = 0,1,2,..,{{d/2}-1}$, such that every two dimensions $j$ and $j+1$ will have their own rotation angles. The formula for theta is the following:

$$
\theta_i = 10000^{-2i/d}
$$

Note that the 1st couple of dimensions, $j=1$, will have a much greater rotation factor that the last, $j=d$. The specific value of $\theta_i$ is a hyperparameter but the authors of RoPE decided to use the same value from the sinusoidal frequencies in the *Attention is All You Need* paper. 

The most efficient way to compute this is by computing $cos{m\theta_i}$ and $sin{m\theta_i}$ once in the forward pass, and just apply it to every single attention block with the following operation:

$$
\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
\vdots \\
x_{d-1} \\
x_d
\end{pmatrix}
\otimes
\begin{pmatrix}
\cos m\theta_1 \\
\cos m\theta_1 \\
\cos m\theta_2 \\
\cos m\theta_2 \\
\vdots \\
\cos m\theta_{d/2} \\
\cos m\theta_{d/2}
\end{pmatrix}
+
\begin{pmatrix}
-x_2 \\
x_1 \\
-x_4 \\
x_3 \\
\vdots \\
-x_d \\
x_{d-1}
\end{pmatrix}
\otimes
\begin{pmatrix}
\sin m\theta_1 \\
\sin m\theta_1 \\
\sin m\theta_2 \\
\sin m\theta_2 \\
\vdots \\
\sin m\theta_{d/2} \\
\sin m\theta_{d/2}
\end{pmatrix}
$$

## [SwiGLU FFN](https://arxiv.org/pdf/2002.05202)

A SwiGLU FFN is an alternative to the feed-forward network proposed by the *Attention is All You Need Paper*. The original version used:

$$FFN(x,W_1, W_2, b_1, b_2) = ReLU(xW_1 + b_1)W_2 + b_2$$

And some more recent variants turned it into by simply ignoring the bias terms:

$$FFN(x,W_1, W_2) = ReLU(xW_1)W_2$$

SwiGLU replaces the ReLU activation function with a GLU(Gated Linear Unit), which consists of the element-wise multiplication of two linear projections where one of them is subject to an activation function $f$:

$$GLU(x,W,V) = f(xW) \odot xV$$

and SwiGLU is a Gated Linear Unit that uses $SiLU_\beta(x) = x\sigma(\beta x)$:

$$FFN_{SwiGLU}(x,W,V,W_2) = (SiLU(xW) \odot xV)W_2$$

It can be visually displayed as the following:
```mermaid
flowchart LR
    x["Input x"] --> w1["Linear projection<br/>xW"]
    x --> v["Linear projection<br/>xV"]
    w1 --> silu["SiLU<br/>σ(xW) · xW"]
    silu --> multiply["Element-wise multiplication<br/>SiLU(xW) ⊙ (xV)"]
    v --> multiply
    multiply --> w2["Output projection<br/>(SiLU(xW) ⊙ xV)W₂"]

    classDef input fill:#1e1b4b,stroke:#a5b4fc,stroke-width:2px,color:#f8fafc
    classDef projection fill:#042f2e,stroke:#5eead4,stroke-width:2px,color:#f8fafc
    classDef activation fill:#2e1065,stroke:#c4b5fd,stroke-width:2px,color:#f8fafc
    classDef operation fill:#431407,stroke:#fdba74,stroke-width:2px,color:#f8fafc
    classDef output fill:#052e16,stroke:#86efac,stroke-width:2px,color:#f8fafc

    class x input
    class w1,v projection
    class silu activation
    class multiply operation
    class w2 output
```
*Note: The linked SwiGLU paper is very interesting because it shows how Deep Learning and LLM development sometimes doesn't a strong governing set of rules behind it, but it has gotten very far by trying different techniques and observing the empirical evidence* 


## [RMSNorm](https://arxiv.org/pdf/1910.07467)

In LayerNorm. We take the output of the linear layer:

$$a_i = \sum^n_{j=1}{w_{ij}x_j}$$

and we apply a normalization where we center by the mean and scale it by the variance:

$$\tilde a_i = \frac{a_i - \mu}{\sigma}$$

The main idea behind that is reduce the effects of *internal covariance shift* in DeepLearning and increase trainning stability. I won't get into specifics, but the authors of the RMSNorm paper propose an alternative that doesn't apply mean centering, claiming their solution achieves similar performance while significantly reducing the computations necessary.
The alternative: 

$$a_i = \frac{a_i}{RMS(a)} $$

where

$$RMS(a) = \sqrt{\frac{1}{n} \sum_{i=1}^n{a_i^2}}$$












