# API Server Hotspot Optimization Options

## How To Use This Reference

Read this file after the profile has already identified the API server bottleneck stages and leaf hotspots.

Recommended reading order:

1. confirm the slow `stage_spans`
2. confirm the leaf hotspot function
3. read the function's internal logic below
4. choose an optimization option based on your constraints

This reference is intentionally option-oriented:

- it explains what the hotspot does internally
- it gives multiple optimization directions
- it keeps trade-offs explicit
- it stays focused on API server work, not engine core or TP worker work

## 1. `Qwen2VLImageProcessorFast._preprocess`

### What it does internally

This is usually the biggest part of `api_stage.hf_processor_wall` for Qwen3.5 image requests.

The main steps are:

1. group images by shape
2. compute `smart_resize`
3. resize tensors
4. rescale and normalize
5. pad temporal dimension when needed
6. reshape and permute into patch layout
7. flatten patches and concatenate outputs
8. produce `pixel_values` and `image_grid_thw`

This means the cost is not just image resize. It is a combined cost of image transforms, memory layout changes, and tensor assembly.

### Good optimization options

- Option A, lowest risk: pre-resize or pre-normalize images before they reach the API server so the HF processor has less work to do.
- Option B, medium risk: add a same-shape fast path so homogeneous image batches avoid repeated group and reorder work.
- Option C, medium to high risk: fuse resize, normalize, and patch flatten work more aggressively to reduce intermediate tensors and memory movement.
- Option D, quality-throughput trade-off: lower effective visual input cost by reducing pixels, image count, or other vision-side limits.

### When to prefer each option

- Prefer A when data preparation is under your control and semantic equivalence matters.
- Prefer B when the request mix often contains same-size images such as fixed benchmark inputs.
- Prefer C when API server CPU time is the priority and you can afford implementation complexity.
- Prefer D only when latency matters more than preserving identical visual fidelity.

## 2. `ProcessorInputs.get_mm_hashes`

### What it does internally

This function computes multimodal cache keys before the processor cache can be used.

The main steps are:

1. iterate modality by modality
2. iterate item by item
3. decide whether to use a provided UUID or compute a fresh hash
4. call `MultiModalHasher.hash_kwargs`
5. recursively serialize the item content
6. hash image bytes, pixel data, tensors, arrays, and processor kwargs

If the item is a `PIL.Image`, the path can reach `np.asarray(obj)` and hash image content, which is expensive for many images.

### Good optimization options

- Option A, lowest risk: if the real workload has very low media reuse, disable or bypass multimodal processor cache so this hash work disappears.
- Option B, low to medium risk: keep cache enabled but provide stable media identities such as `mm_uuid_items`, EXIF image IDs, or upstream media IDs.
- Option C, medium risk: add a cheaper fast path for file or byte-backed media so hashing can use stable source bytes instead of reconstructed pixel arrays.
- Option D, high risk: redefine cache-key policy around caller-provided identity plus processor kwargs instead of full content hashing.

### When to prefer each option

- Prefer A when benchmark or production inputs are usually unique and cache hits are rare.
- Prefer B when cache reuse is valuable but upstream can provide stable identities.
- Prefer C when you need cache reuse and can still preserve content-equivalent semantics.
- Prefer D only when you are ready to explicitly own cache-key correctness rules.

## 3. `SingleWriterShmObjectStorage.copy_to_buffer`

### What it does internally

This function is the write-side data copy step for the shared-memory multimodal cache.

The main steps are:

1. `put()` serializes the object first
2. buffer memory is allocated from the SHM ring buffer
3. metadata is copied into the destination memoryview
4. payload bytes are copied into the same memoryview
5. for `list[bytes]`, copy happens item by item in a Python loop

The important interpretation detail is:

- `aggregate-total` can be smaller than `wall-critical-path`

That happens when many small copy calls are spread across a wider first-start to last-end window with gaps between them.

### Good optimization options

- Option A, lowest risk: if multimodal SHM cache is not worth keeping, remove the need for these writes by disabling or bypassing that cache path.
- Option B, medium risk: reduce segment count so each write is a larger contiguous copy instead of many small copy operations.
- Option C, medium to high risk: batch multiple multimodal outputs into fewer SHM writes at request scope instead of item scope.
- Option D, high risk: redesign serialization and SHM handoff for fewer copies or near zero-copy access.

### When to prefer each option

- Prefer A when `get_mm_hashes` is also expensive and cache reuse is weak.
- Prefer B when cache must stay but the current serializer returns many small payload segments.
- Prefer C when the workload regularly sends many images per request.
- Prefer D only when SHM handoff is a long-term architecture target.

## 4. `MediaConnector.load_from_url_async`

### What it does internally

This function is usually a concurrent media loading stage rather than a single-thread sequential bottleneck.

The main steps are:

1. parse the URL
2. choose HTTP, data URL, or file path branch
3. fetch bytes
4. dispatch decode or load work into the thread pool
5. return the decoded media object

For many-image requests, the key metric is usually `concurrency_windows.window_wall_ms`, not `aggregate_total_ms`.

### Good optimization options

- Option A, lowest risk: keep data local so file and bytes access stay cheap.
- Option B, medium risk: add faster local-file paths or dedicated executors for decode work.
- Option C, medium risk: if URLs are remote HTTP, optimize connection reuse, concurrency, and prefetch behavior.

### When to prefer each option

- Prefer A for local benchmark datasets and controlled media roots.
- Prefer B when file loading and decode start to occupy visible wall-critical-path time.
- Prefer C only when the real workload actually depends on remote URLs.

## 5. `Qwen2TokenizerFast.__call__` and `AsyncMicrobatchTokenizer.encode`

### What it does internally

This part is usually not the first bottleneck for the current multimodal benchmark, but it still matters after vision-side work shrinks.

The main steps are:

1. `Qwen3VLProcessor.__call__` expands image placeholders in text
2. `Qwen2TokenizerFast.__call__` normalizes arguments and dispatches into fast tokenizer encode flow
3. the Rust-backed tokenizer performs actual subword tokenization
4. `AsyncMicrobatchTokenizer.encode` can batch multiple pending encode requests when concurrency exists

### Good optimization options

- Option A, lowest risk: cache repeated prefixes or repeated prompt templates when the workload has high text reuse.
- Option B, medium risk: optimize placeholder expansion so the API server does less repeated large-string work before tokenization.
- Option C, medium risk: rely more on microbatch tokenization when the deployment has real concurrent requests.

### When to prefer each option

- Prefer A when there is real prompt reuse.
- Prefer B when placeholder expansion itself becomes measurable in the API server stage.
- Prefer C when the service handles multiple requests concurrently instead of one single benchmark request.

## 6. Wrapper Functions And How To Use Them

These functions are useful stage anchors, but they should not be the first optimization target:

- `InputProcessingContext.call_hf_processor`
- `Qwen3VLMultiModalProcessor._call_hf_processor`
- `Qwen3VLProcessor.__call__`

Use them for:

- stage location
- tracing where time moves after an optimization
- deciding which leaf hotspot to inspect next

Do not use them for:

- final hotspot conclusions
- direct “optimize this giant wrapper” instructions without sub-step analysis

## Default Prioritization

If the current profile looks similar to the observed Qwen3.5 API server case, a practical default order is:

1. decide whether multimodal cache is worth keeping
2. optimize `Qwen2VLImageProcessorFast._preprocess`
3. optimize `SingleWriterShmObjectStorage.copy_to_buffer` if cache remains
4. optimize `ProcessorInputs.get_mm_hashes` if cache remains
5. revisit tokenizer and media load only after the first three are under control

## Output Style Recommendation

When answering a user who wants optimization guidance, format the answer per hotspot:

1. function name
2. internal execution logic
3. why it is or is not on the critical path
4. multiple optimization options
5. trade-offs and when to pick each option

If the user says “only give options first”, stop before implementation.
