# vLLM and vLLM-Ascend Architecture

## Scope

This note focuses on the stock architecture of upstream `vllm` plus the `vllm-ascend` hardware plugin. It does not assume out-of-tree patches or local phase experiments.

## 1. What vLLM is responsible for

At a high level, `vllm` is the serving engine. It provides:

- An OpenAI-compatible HTTP server.
- The request scheduler.
- KV cache management.
- Tokenization and output streaming.
- Distributed execution over TP, PP, and DP workers.
- Multimodal request plumbing for text plus image, video, or audio inputs.

In V1, `vllm` uses a multi-process design. The important processes are:

- API server
  Accepts HTTP requests, parses OpenAI-compatible payloads, tokenizes text, handles multimodal input loading, and streams responses.
- Engine core
  Owns scheduling and KV cache decisions. It decides how many tokens to advance per request and when to dispatch work.
- Worker processes
  One worker per accelerator device. A worker owns the model runner, model weights, device memory, and per-step execution.

For a typical `TP=4` single-node service, the common shape is:

- 1 API server
- 1 engine core
- 4 worker processes

That is the core mental model to keep when debugging behavior or performance.

## 2. The standard request path

### Text-only path

The steady-state path is:

1. Client sends a request to the API server.
2. API server validates the request and tokenizes the prompt.
3. API server sends the request to the engine core.
4. Engine core schedules a step and dispatches work to workers.
5. Workers run the forward pass and return sampled tokens.
6. API server streams tokens back to the client.

### Multimodal path

For multimodal models, the path adds media work:

1. API server parses the OpenAI-compatible message content.
2. Media references are resolved or fetched.
3. Multimodal processor logic prepares media inputs for the model.
4. Engine core schedules the request.
5. Workers run the vision encoder and language model path.
6. The server streams decoded text back to the client.

The expensive parts usually move between three places:

- Media loading and path resolution on the API side.
- Multimodal preprocessing before the vision encoder.
- The vision encoder itself on workers.

## 3. Why V1 matters

V1 changes the way vLLM organizes the serving core:

- It uses a unified scheduler instead of a strict prefill-vs-decode split.
- Chunked prefill is enabled by default when possible.
- Prefix caching, speculative decoding, and other optimizations are designed into the same scheduler model.
- More features are enabled automatically instead of requiring many knobs.

Two consequences matter in practice:

- If you only omit a Boolean scheduler flag, you may still get the V1 default behavior.
- Reproducibility and performance both depend on the exact runtime configuration, not only on request payloads.

## 4. Where vllm-ascend fits

`vllm-ascend` is not a forked serving entrypoint. It is a hardware plugin that adapts upstream `vllm` to Ascend NPU.

Think of the split like this:

- `vllm`
  Owns the serving framework and model-serving abstractions.
- `vllm-ascend`
  Provides the Ascend platform integration: platform registration, worker behavior, custom operators, device communication, compilation hooks, and Ascend-specific performance features.

The important architectural point is that `vllm-ascend` plugs into upstream `vllm`; it does not replace the whole serving stack.

## 5. How vllm-ascend integrates

`vllm-ascend` follows the vLLM hardware-plugin model:

- Platform plugin registration tells vLLM that the Ascend platform is available.
- Platform-specific configuration updates choose Ascend worker classes, attention backends, device communicators, and related settings.
- Worker-side behavior is extended through inheritance and NPU-specific components.
- When necessary, `vllm-ascend` patches targeted upstream components to adapt model behavior or runtime behavior for Ascend.
- Custom operators and communication backends supply hardware-specific acceleration.

From a practical operator's perspective, this means:

- You still launch the service with the normal `vllm serve` interface.
- If the plugin is installed correctly and the environment is supported, Ascend-specific classes are selected automatically.
- The service logs usually reveal that the Ascend platform plugin was discovered and activated.

## 6. Ascend-specific execution pieces

### torch-npu and CANN

The low-level Ascend runtime stack is exposed through `torch-npu` and CANN. `vllm-ascend` builds on top of that stack rather than bypassing it.

### HCCL

HCCL is the collective communication layer on Ascend. In practice it matters for:

- Tensor-parallel communication.
- Data-parallel communication.
- Some NPU-side distributed loading or weight movement features.

Environment variables such as `HCCL_BUFFSIZE` influence memory reservation and communication behavior, so they should be treated as part of the serving configuration, not an afterthought.

### Worker and model runner changes

The Ascend worker path is where the hardware-specific behavior actually lands:

- Device initialization
- Model loading
- KV cache initialization
- Step execution
- Communication backend choice
- NPU-specific graph or custom-op paths

That is why many performance or correctness issues on Ascend eventually show up as worker-side problems even when the request entered through the standard API server.

## 7. Multimodal serving details that matter on Ascend

For VL models such as Qwen-VL style models, these controls are especially important:

- `--allowed-local-media-path`
  Required when the request uses local files through `file://` URLs.
- `--mm-processor-cache-type shm`
  Makes the multimodal processor cache use shared memory so multi-process serving can reuse cached artifacts more efficiently.
- `--mm-processor-cache-gb`
  Controls the SHM cache size budget.
- `--mm-encoder-tp-mode data`
  Makes the multimodal encoder run in a data-parallel style across TP workers, which is often the practical choice for vision-heavy requests.

These are not random tuning flags. They change where the cost lands and how multimodal preprocessing interacts with the worker topology.

## 8. Why eager mode is the safest baseline

Many Ascend guides, debugging flows, and profiling flows use `--enforce-eager`. The reasons are pragmatic:

- Fewer graph-related surprises during bring-up.
- Easier stack traces and profiling.
- More predictable behavior while validating a new service path.

It is not always the fastest possible mode, but it is the cleanest baseline when the goal is to bring up, benchmark, or validate a service in a controlled way.

## 9. Mental model for debugging

When something goes wrong, use this order:

1. Environment and version alignment
   Confirm `vllm` and `vllm-ascend` versions match and Ascend runtime is sourced.
2. Platform activation
   Confirm the Ascend plugin is actually selected at startup.
3. API-side request shape
   Confirm the payload is valid, the model path is correct, and local media is allowed.
4. Engine core scheduling
   Confirm the model length, TP size, and scheduling flags are sane.
5. Worker execution
   Confirm model loading, HCCL setup, memory headroom, and multimodal flags.

This is usually faster than starting from model internals.
