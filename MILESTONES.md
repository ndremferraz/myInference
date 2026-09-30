# JAX LLM Inference Engine — Milestones

## End Goal

Build a JAX-based LLM inference engine capable of serving open-source models such as Gemma and DeepSeek-family models across a multi-host TPU slice / TPU Pod on Google Cloud.

---
## Milestone 1 — Tiny Transformer in Pure JAX

### Goal
Implement a complete decoder-only Transformer forward pass. Use [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) as reference.

### Build
- [ ] MLP
- [ ] Attention
- [ ] Transformer block
- [ ] Multi-layer Transformer
- [ ] LM head
- [ ] Downloading Model heights
- [ ] Greedy decoding
- [ ] Temperature sampling
- [ ] Top-k / top-p sampling

### Validate
- [ ] Compare intermediate tensor shapes with a reference model
- [ ] Verify deterministic greedy generation

### Done When
Given a prompt, your implementation generates coherent text on one accelerator and its logits closely match a trusted reference implementation.

---

## Milestone 2 — Prefill, Decode, and KV Cache

### Goal
Turn the Transformer implementation into an actual autoregressive inference runtime.

### Build
- [ ] Separate prefill execution path
- [ ] Separate decode execution path
- [ ] Preallocated KV cache
- [ ] KV-cache indexing
- [ ] Per-layer K/V storage
- [ ] Sequence position tracking
- [ ] Token-by-token generation loop

### Benchmark
Measure at several sequence lengths, for example:

- [ ] 128 tokens
- [ ] 1K tokens
- [ ] 4K tokens
- [ ] 16K tokens

Track:

- [ ] Prefill tokens/sec
- [ ] Decode tokens/sec
- [ ] Time to first token (TTFT)
- [ ] Time per output token (TPOT)
- [ ] KV-cache memory usage
- [ ] Total accelerator memory usage

### Done When
Your engine generates efficiently using a preallocated KV cache, and you can predict how increasing sequence length affects memory consumption and latency.

---

## Milestone 3 — Paged KV Cache

### Goal
Build a KV-cache memory manager suitable for variable-length concurrent workloads.

### Build
- [ ] Fixed-size physical KV blocks
- [ ] Logical sequence blocks
- [ ] Block tables
- [ ] Block allocator
- [ ] Block freeing
- [ ] Sequence-to-block mapping
- [ ] Cache capacity accounting
- [ ] Out-of-memory handling

### Explore
- [ ] Internal fragmentation
- [ ] External fragmentation
- [ ] Block-size tradeoffs
- [ ] Prefix sharing
- [ ] Prefix caching

### Done When
Many variable-length sequences can coexist without requiring a separate giant contiguous KV allocation for every request.

---

## Milestone 4 — Multi-Device TPU Inference

### Goal
Run the model across multiple TPU devices and understand every major collective operation involved.

### Learn
- [ ] JAX `Mesh`
- [ ] Named sharding
- [ ] `PartitionSpec`
- [ ] `shard_map`
- [ ] Device-local tensor shapes
- [ ] Replicated vs. sharded tensors

### Implement
- [ ] Sharded model weights
- [ ] Tensor-parallel attention
- [ ] Tensor-parallel MLP
- [ ] Sharded KV cache where appropriate

### Understand Collectives
- [ ] All-reduce
- [ ] All-gather
- [ ] Reduce-scatter
- [ ] All-to-all

### Scale
- [ ] 2 devices
- [ ] 4 devices
- [ ] 8 devices

### Done When
The model runs correctly across multiple TPU devices and you can explain why every significant collective communication operation exists.

---

# Recommended Execution Order

```text
[ ] 1. Tiny Transformer in pure JAX
[ ] 2. Prefill + decode + KV cache
[ ] 3. Paged KV cache
[ ] 4. Multi-device TPU inference
```


