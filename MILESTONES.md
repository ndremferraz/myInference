# JAX LLM Inference Engine — Milestones

## End Goal

Build a JAX-based LLM inference engine capable of serving open-source models such as Gemma and DeepSeek-family models across a multi-host TPU slice / TPU Pod on Google Cloud.

The project should progress from correctness on a single device to efficient distributed inference. Avoid hiding the important systems concepts behind high-level serving frameworks while learning.

---

## Milestone 0 — Foundations

### Goal
Build the mental model required to reason about LLM inference performance.

### Understand
- [ ] Decoder-only Transformer architecture
- [ ] Multi-Head Attention (MHA)
- [ ] Grouped-Query Attention (GQA)
- [ ] Multi-Query Attention (MQA)
- [ ] RoPE
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] Autoregressive decoding
- [ ] Prefill vs. decode
- [ ] KV caching
- [ ] BF16, FP16, FP8, INT8 concepts
- [ ] Arithmetic intensity
- [ ] Roofline model
- [ ] Compute-bound vs. memory-bound operations
- [ ] TPU HBM and MXUs
- [ ] TPU interconnect / ICI
- [ ] JAX tracing and `jit`
- [ ] PyTrees
- [ ] Static shapes and recompilation
- [ ] XLA basics
- [ ] JAX device meshes and sharding

### Done When
You can explain why LLM prefill and decode have different performance characteristics and identify the main compute, memory, and communication bottlenecks in each.

---

## Milestone 1 — Tiny Transformer in Pure JAX

### Goal
Implement a complete decoder-only Transformer forward pass without relying on Flax for the model implementation.

### Build
- [ ] Token embeddings
- [ ] RMSNorm
- [ ] RoPE
- [ ] Attention
- [ ] GQA
- [ ] SwiGLU MLP
- [ ] Transformer block
- [ ] Multi-layer Transformer
- [ ] LM head
- [ ] Greedy decoding
- [ ] Temperature sampling
- [ ] Top-k / top-p sampling
- [ ] Basic checkpoint/weight loading

### Validate
- [ ] Compare intermediate tensor shapes with a reference model
- [ ] Compare final logits with a reference implementation
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

## Milestone 3 — Understand JAX/XLA Execution

### Goal
Understand what JAX actually compiles and why your implementation performs the way it does.

### Investigate
- [ ] JAX tracing
- [ ] `jaxpr`
- [ ] StableHLO
- [ ] XLA compilation
- [ ] TPU executables
- [ ] Static vs. dynamic shapes
- [ ] Recompilation behavior
- [ ] Operator fusion
- [ ] Buffer donation
- [ ] Asynchronous execution

### Profile
- [ ] Prefill
- [ ] Decode
- [ ] Attention
- [ ] MLP
- [ ] KV-cache reads/writes

### Done When
For a performance regression, you can determine whether the primary cause is compute, HBM bandwidth, tensor layout, recompilation, synchronization, or communication.

---

## Milestone 4 — Static Batching

### Goal
Understand how batching changes accelerator utilization and inference economics.

### Implement
- [ ] Batched prefill
- [ ] Batched decode
- [ ] Per-sequence positions
- [ ] Per-sequence sampling state
- [ ] Padding / masking

### Benchmark
Test batch sizes such as:

- [ ] 1
- [ ] 2
- [ ] 4
- [ ] 8
- [ ] 16
- [ ] 32

For each, record:

- [ ] Total tokens/sec
- [ ] Per-user latency
- [ ] TTFT
- [ ] TPOT
- [ ] Memory consumption
- [ ] Accelerator utilization

### Done When
You can explain where increased batch size improves throughput, where latency begins to degrade, and why.

---

## Milestone 5 — Continuous Batching and Scheduler

### Goal
Move from a model runner to an inference engine capable of handling independent concurrent requests.

### Build
- [ ] Request abstraction
- [ ] Request queue
- [ ] Sequence state
- [ ] Scheduler
- [ ] Decode slots
- [ ] Dynamic insertion of new requests
- [ ] Removal of completed requests
- [ ] Sampling state per request
- [ ] Cancellation handling
- [ ] Maximum sequence-length enforcement

### Done When
Requests can arrive and finish independently while the accelerator continuously processes active sequences without waiting for an entire static batch to finish.

---

## Milestone 6 — Paged KV Cache

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

## Milestone 7 — Multi-Device TPU Inference

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

## Milestone 8 — Parallelism and Sharding Strategy

### Goal
Learn to map model architecture onto accelerator topology instead of blindly choosing a sharding configuration.

### Explore
- [ ] Tensor parallelism (TP)
- [ ] Data parallelism (DP)
- [ ] Pipeline parallelism (PP)
- [ ] Expert parallelism (EP)

### Model
For a proposed configuration, estimate:

- [ ] Weight memory/device
- [ ] KV-cache memory/device
- [ ] Activation memory/device
- [ ] FLOPs/device
- [ ] Communication volume
- [ ] Expected communication frequency

### Done When
Given model dimensions and a TPU topology, you can propose a sharding strategy and explain the compute, memory, and communication tradeoffs before running it.

---

## Milestone 9 — Custom TPU Kernels with Pallas

### Goal
Move below ordinary JAX/XLA abstractions and understand TPU kernel execution and data movement.

### Learn
- [ ] Pallas programming model
- [ ] HBM
- [ ] VMEM
- [ ] DMA
- [ ] Tiling
- [ ] Pipelining
- [ ] Double buffering
- [ ] TPU matrix operations
- [ ] Compute/data-transfer overlap

### Kernel Progression
- [ ] Simple elementwise operation
- [ ] Matrix multiplication
- [ ] RMSNorm
- [ ] RoPE
- [ ] Attention-related kernel
- [ ] Fused inference operation

### Compare
For each useful custom kernel:

- [ ] Correctness vs. JAX implementation
- [ ] Latency
- [ ] Throughput
- [ ] HBM traffic
- [ ] Compilation behavior

### Done When
At least one important inference path uses a kernel you wrote yourself, and you understand why its performance differs from the XLA-generated version.

---

## Milestone 10 — Gemma Integration

### Goal
Run a real Gemma checkpoint entirely through your inference engine.

### Build
- [ ] Gemma configuration parser
- [ ] Checkpoint loader
- [ ] Weight-name mapping
- [ ] Weight layout conversion
- [ ] Architecture-specific configuration
- [ ] Tokenizer integration

### Validate
- [ ] Compare logits against a trusted Gemma implementation
- [ ] Compare greedy generations
- [ ] Validate long-context KV-cache behavior
- [ ] Benchmark prefill
- [ ] Benchmark decode

### Progression
- [ ] Start with a small Gemma checkpoint
- [ ] Move to a medium-sized checkpoint
- [ ] Test a model requiring multi-device inference

### Done When
An official Gemma checkpoint generates correct outputs through your own model implementation, KV-cache system, scheduler, and execution runtime.

---

## Milestone 11 — Mixture-of-Experts Runtime

### Goal
Extend the engine beyond dense Transformer models.

### Learn / Build
- [ ] MoE layer
- [ ] Router
- [ ] Top-k expert selection
- [ ] Token-to-expert dispatch
- [ ] Expert execution
- [ ] Expert output combination
- [ ] Expert parallelism
- [ ] All-to-all communication
- [ ] Load imbalance measurement
- [ ] Expert capacity management

### Done When
A smaller open MoE model runs correctly and efficiently across multiple devices.

---

## Milestone 12 — DeepSeek-Family Architecture

### Goal
Support the architectural features required by a DeepSeek-family model rather than treating it as a generic dense Transformer.

### Study / Implement
- [ ] DeepSeek model configuration
- [ ] MLA-style attention concepts where required by the target checkpoint
- [ ] DeepSeek RoPE/position handling
- [ ] MoE architecture
- [ ] Routing behavior
- [ ] Shared/routed experts where applicable
- [ ] Expert parallelism strategy
- [ ] KV-cache implications
- [ ] Checkpoint conversion

### Validate
- [ ] Reference logits
- [ ] Greedy generation
- [ ] Multi-device execution
- [ ] MoE communication profile

### Done When
A target DeepSeek-family checkpoint runs through your engine with validated output correctness.

---

## Milestone 13 — Multi-Host TPU

### Goal
Move from a single TPU host to distributed multi-host execution.

### Build / Understand
- [ ] JAX distributed initialization
- [ ] Global device mesh
- [ ] Process/device indexing
- [ ] Multi-host checkpoint loading
- [ ] Cross-host collectives
- [ ] TPU topology
- [ ] ICI-aware sharding
- [ ] Synchronization behavior
- [ ] Failure behavior

### Scale
- [ ] Single host
- [ ] Two hosts
- [ ] Small multi-host slice
- [ ] Larger TPU slice

### Measure
- [ ] Single-device baseline
- [ ] Multi-device scaling
- [ ] Multi-host scaling
- [ ] Communication overhead
- [ ] Scaling efficiency

### Done When
A large model runs across multiple TPU hosts and you can quantify where scaling efficiency is being lost.

---

## Milestone 14 — TPU Pod / Large-Slice Optimization

### Goal
Optimize the complete engine for large distributed TPU inference.

### Optimize
- [ ] TP topology mapping
- [ ] PP topology mapping
- [ ] EP topology mapping
- [ ] DP replica placement
- [ ] Collective communication
- [ ] Communication/computation overlap
- [ ] KV-cache placement
- [ ] Batch scheduling
- [ ] Prefill/decode balance

### Done When
You can choose a TPU topology and parallelism configuration for a large model based on measured memory, compute, and communication constraints rather than trial and error.

---

## Milestone 15 — Production Serving Layer

### Goal
Turn the inference runtime into a usable serving system.

### Build
- [ ] HTTP or gRPC API
- [ ] Tokenizer service/path
- [ ] Request admission control
- [ ] Continuous batching scheduler
- [ ] Streaming token responses
- [ ] Request cancellation
- [ ] Timeouts
- [ ] Prefix caching
- [ ] Backpressure
- [ ] Metrics
- [ ] Tracing
- [ ] Logging
- [ ] Health checks
- [ ] Multi-host worker coordination
- [ ] Failure recovery

### Core Metrics
- [ ] Time to first token (TTFT)
- [ ] Time per output token (TPOT)
- [ ] Prefill tokens/sec
- [ ] Decode tokens/sec
- [ ] Total tokens/sec
- [ ] Requests/sec
- [ ] Goodput under latency SLO
- [ ] HBM utilization
- [ ] TPU utilization
- [ ] Cost per million tokens

### Done When
The engine can accept concurrent external requests, continuously batch them, stream results, recover from expected failures, and expose enough telemetry to diagnose performance.

---

# Final Completion Criteria

The project is complete when you can take an open-source checkpoint and control the full path:

```text
Checkpoint
    ↓
Weight Conversion
    ↓
JAX Model
    ↓
Sharded Parameters
    ↓
Prefill / Decode Runtime
    ↓
KV Cache Manager
    ↓
Continuous-Batching Scheduler
    ↓
TP / PP / EP / DP
    ↓
XLA + Custom Pallas Kernels
    ↓
Multi-Host TPU Execution
    ↓
Serving Layer
    ↓
Streaming Tokens
```

You should be able to explain and measure the major costs at every layer: **compute, HBM traffic, KV-cache capacity, communication, compilation, scheduling, latency, throughput, and infrastructure cost.**

---

# Recommended Execution Order

```text
[ ] 0. Foundations
[ ] 1. Tiny Transformer in pure JAX
[ ] 2. Prefill + decode + KV cache
[ ] 3. JAX/XLA profiling
[ ] 4. Static batching
[ ] 5. Continuous batching + scheduler
[ ] 6. Paged KV cache
[ ] 7. Multi-device TPU inference
[ ] 8. Sharding / parallelism strategy
[ ] 9. Pallas kernels
[ ] 10. Gemma integration
[ ] 11. MoE runtime
[ ] 12. DeepSeek-family integration
[ ] 13. Multi-host TPU
[ ] 14. TPU Pod / large-slice optimization
[ ] 15. Production serving layer
```

## Rule for Moving Forward

Do not advance simply because the previous milestone runs. Advance when you can **explain why it works, measure its performance, identify its bottlenecks, and predict how it will behave when scaled.**
