# model-to-webgpu Methodology

Convert any ML model into a standalone browser project with hand-written WGSL compute shaders. No runtime framework, no ONNX Runtime, no transformers.js. The output is a zero-dependency TypeScript package that runs inference entirely via WebGPU compute dispatches.

## What This Methodology Produces

Given a model file (ONNX, GGUF, or TFLite), this methodology guides you through generating a complete standalone project:

```
{model-name}-webgpu/
  src/
    types.ts           # Config interfaces, buffer types, bind group cache types
    engine.ts          # Forward pass orchestrator — chains shader dispatches
    shaders.ts         # All WGSL compute shaders as string literals
    weights.ts         # Weight loader — format-specific parser + dequantization
    tokenizer.ts       # Preprocessing (BPE tokenizer, phonemizer, image resize)
    index.ts           # Public API (loadModel, generate/infer)
  scripts/
    inspect_model.py   # Graph inspector — maps ops, weights, quantization
    dump_activations.py # Reference activation dumper — ground truth for validation
    extract_weights.py  # Weight extraction + format conversion (ONNX/TFLite only)
  tests/
    activation-match.spec.ts  # Playwright test — WebGPU vs reference comparison
  models/
    activations/       # Reference .npy files from dump_activations.py
    weights/           # Extracted weight binaries + JSON manifests
  package.json         # Zero dependencies, tsup build
  tsconfig.json
```

The output is NOT a generic runtime. Every generated file is model-specific, readable, and modifiable. The shaders are hand-written WGSL — not auto-generated from a graph compiler. Each shader is written understanding the exact tensor shapes, strides, and data flow for the specific model.

## When to Use

- Running an ML model in the browser via WebGPU
- Porting an existing model (PyTorch, ONNX, TFLite, GGUF) to pure WebGPU
- When you need hand-written shaders, zero dependencies, minimal bundle size
- Building browser-native ML inference

## When NOT to Use

Use ONNX Runtime Web or Transformers.js instead when:
- **Prototyping speed matters more than bundle size** — frameworks get you running in hours, not days
- **Model uses standard ops only** — no custom activations, no unusual quantization. Frameworks handle standard graphs fine.
- **Bundle size is irrelevant** — if you are building an internal tool, the 2-5MB framework overhead does not matter
- **Model is >1B parameters** — too large for mobile WebGPU anyway; server-side inference is better
- **You need broad hardware compatibility** — hand-written shaders optimized for Apple GPUs may underperform on Intel/AMD iGPUs

---

## Phases

This methodology operates in 9 sequential phases. Each phase has clear inputs, outputs, and success criteria. You can enter any phase independently — just ensure its prerequisites are met.

### Phase 1: Inspect

**Goal:** Understand the model's graph topology, operations inventory, weight layout, and quantization scheme. Build a mental model of the full inference pipeline before writing any code.

**Inputs:** A model file (`.onnx`, `.gguf`, `.tflite`) or a HuggingFace model ID.

**Steps:**

1. **Download the model** if given a HuggingFace ID. For GGUF, prefer Q8_0 quantization. For ONNX, prefer the quantized variant if available.

   > **WARNING (TFLite):** Check if the model is inside a `.task` zip bundle (MediaPipe convention — extract the `.tflite` from the zip). Also: MediaPipe models come in LITE and FULL variants with **completely different architectures, weight counts, layer names, and even activation functions.** Always verify which variant the reference runtime uses. Picking the wrong one wastes days — you'll get plausible but wrong output and spend hours debugging "accuracy issues" that are actually a different model.

2. **Write `scripts/inspect_model.py`** — a Python script that:
   - Parses the model file (onnxruntime for ONNX, struct/custom parser for GGUF, tensorflow for TFLite)
   - Lists all operations/nodes grouped by type with counts
   - Lists all weight tensors with: name, shape, dtype, quantization type, byte size
   - Identifies the model's named components/blocks (encoder, decoder, attention layers, etc.)
   - Maps the forward-pass data flow: input → ... → output
   - Detects quantization scheme: which ops are quantized, what format (INT8, UINT8, Q8_0, Q4_0, float16)
   - **For multi-model architectures** (e.g., detector + landmark model): inspect EACH model separately and document the inter-model data flow (what detector output becomes landmark input)
   - Outputs a structured summary

   See `templates/inspect_onnx.py`, `templates/inspect_tflite.py`, and `templates/inspect_gguf.py` for starter templates.

3. **Identify the preprocessing pipeline:**
   - LLM: tokenizer type (BPE, SentencePiece, etc.), special tokens, chat template format
   - TTS: phonemizer (espeak-ng, dictionary lookup), text normalization rules
   - Vision: input resolution, normalization (mean/std or [0,1]), letterboxing/cropping strategy
   - Note whether preprocessing can be done in pure JS or requires WASM (important for Safari compatibility)

4. **Document findings** in the project's architecture doc under "Model Architecture":
   - Total parameter count and weight size
   - Named components with their op composition
   - Quantization strategy
   - Input/output tensor shapes and semantics
   - Preprocessing pipeline requirements
   - Any non-standard ops that will need special shaders (DynamicQuantizeLSTM, Snake activation, PReLU, etc.)
   - For multi-model: full pipeline diagram showing model A → glue logic → model B
   - Tied weights (e.g., Gemma's embedding = LM head) — note these to avoid duplicating GPU buffers

**Success criteria:**
- Every operation type in the model is identified and mapped to a component
- Weight tensor count, shapes, and quantization types are fully catalogued
- Forward-pass data flow is documented as a pipeline diagram
- Non-standard ops are flagged with notes on how to implement them
- Preprocessing requirements are documented
- For multi-model: inter-model data flow is documented

**Key lessons learned:**
- ONNX models may use quantized ops (`ConvInteger`, `MatMulInteger`, `DynamicQuantizeLSTM`) that look different from standard ops — inspect carefully
- GGUF bundles config metadata AND tokenizer vocabulary in the file — extract both
- TFLite batch normalization may be fused into conv bias at export time — check for `FusedBatchNormV3`
- Some projects use TWO models (e.g., palm detection + hand landmark) with NMS and affine-crop glue between them. Not all projects are single-model.

---

### Phase 2: Extract Weights + Preprocessing

**Goal:** Extract all model weights from the source format, apply necessary transformations, and build the preprocessing pipeline (tokenizer, phonemizer, or image preprocessor).

**Inputs:** Model file from Phase 1. Model architecture knowledge from Phase 1's documentation.

**Steps:**

1. **Weight extraction** — two approaches depending on source format:

   **For ONNX and TFLite: Python extraction to binary blob**

   Write `scripts/extract_weights.py` that:
   - Loads the model via its native runtime
   - Iterates all weight tensors
   - Applies format-specific transformations:

   **ONNX transformations:**
   - Dequantize INT8/UINT8 weights to float32: `f32 = (int_val - zero_point) * scale`
   - Handle per-axis quantization (e.g., LSTM weights with `scale[num_directions]`)
   - Convert float16 weights to float32
   - Watch for varint-encoded packed fields in the protobuf — raw int32 arrays may need varint parsing
   - Key lesson: dequantize at load time, use standard f32 matmul/conv shaders. Don't try to implement dynamic quantization on GPU.

   **TFLite transformations:**
   - Transpose NHWC weights to NCHW: regular conv `[O,H,W,I]→[O,I,H,W]`, depthwise `[1,H,W,C]→[C,1,H,W]`
   - Handle fused BatchNorm: if BN params are folded into conv bias at export, each conv layer just needs weight + fused_bias (not 4 separate BN params)
   - Watch for LITE vs FULL model variants — they have completely different weight naming
   - Float16 storage with optional f32 conversion

   Output: binary blob + JSON manifest with keys, shapes, byte offsets, and dtype.

   **For GGUF: TypeScript runtime parser (no Python extraction)**

   GGUF models are parsed at runtime in the browser. Write `src/gguf.ts` that:
   - Reads the GGUF binary header (magic bytes, version, tensor count, KV metadata)
   - Extracts model config from metadata keys (hidden_size, num_layers, head_count, etc.)
   - Extracts tokenizer vocabulary from `tokenizer.ggml.tokens` metadata
   - Builds a tensor info table mapping name → {shape, quantization_type, byte_offset}
   - Loads weights via HTTP Range requests: fetch first 20MB for header, then per-layer Range requests
   - Packs Q8_0 blocks for GPU: 34-byte block (2B f16 scale + 32B int8) → 9 uint32 (1 f32 scale + 8 packed int8)
   - Packs Q4_0 similarly: 18-byte block → 5 uint32
   - **Keep embedding in quantized format on GPU** — dequantizing a 262K-token embedding to f32 uses 671MB and crashes iPhones. Use an `embeddingLookupQ8` shader that dequantizes per-token on the fly.

2. **Build the preprocessing pipeline** — write `src/tokenizer.ts` (or `src/phonemizer.ts`, `src/preprocessor.ts`):

   **For LLMs:**
   - Extract tokenizer from GGUF metadata or separate tokenizer.json
   - Implement BPE or longest-prefix-match tokenization in pure JS
   - Handle special tokens: BOS, EOS, turn markers
   - Build chat template formatter (model-family specific)
   - Multi-turn conversation manager with KV cache position tracking

   **For TTS:**
   - Port phonemizer to pure JS if possible (avoids WASM/Safari issues)
   - Build a pronunciation dictionary (e.g., 234K-word CMU-style dictionary)
   - Implement text normalization (numbers, abbreviations, punctuation)
   - **Safari doesn't support `ReadableStream[Symbol.asyncIterator]`** — if using any streaming/WASM, use manual `getReader()` loop with `while(true) { const {done, value} = await reader.read(); ... }` pattern

   **For Vision:**
   - Implement letterbox resize on GPU (`letterboxResizeShader`) matching the exact coordinate formula the reference model uses
   - Implement affine crop on GPU for multi-model pipelines (detector → crop → landmark model)
   - Match the reference model's exact normalization (e.g., [0,1] vs [-1,1] vs ImageNet mean/std)

3. **Validation:** Compare extracted weight values against the source model: `np.allclose(extracted, original, atol=1e-3)`.

**Success criteria:**
- All weight tensors extracted/parseable and GPU-ready
- Preprocessing pipeline produces output matching the reference runtime
- Architecture doc updated with weight format and preprocessing documentation

**Key lessons learned:**
- ONNX protobuf varint parsing is subtle — packed int32 fields use varint encoding. Getting this wrong causes zero_point=0 instead of zero_point=129, making ALL dequantized weights wrong.
- TFLite weight names differ between LITE and FULL variants — build explicit name-mapping tables.
- Weight key collisions happen (e.g., `conv_landmarks` appearing twice with different shapes) — disambiguate with shape suffix.
- For models >100MB, split weights per-layer for HTTP Range request loading (critical for mobile).

---

### Phase 3: Reference Activations

**Goal:** Generate ground-truth intermediate activations for every pipeline stage by running the model through its native runtime. These are the "oracle" for Phase 6 validation.

**Inputs:** Weights from Phase 2. A representative test input.

**Steps:**

1. **Choose a test input** appropriate for the model type:
   - LLM: A short prompt like `"Hello, how are you?"`
   - TTS: A short sentence like `"Hello, this is a test."`
   - Vision: A test image (solid color or simple pattern for determinism)
   - Record the exact input — it must be reproduced identically in WebGPU

2. **Write `scripts/dump_activations.py`** that:
   - Loads the model via its native runtime (ONNX Runtime, TF Lite, transformers)
   - Runs inference on the test input
   - Hooks into every intermediate computation and saves the output tensor as a `.npy` file
   - Names files by pipeline stage: `01_embedding.npy`, `02_encoder_output.npy`, etc.
   - Saves to `models/activations/`
   - **Save both input AND output** of each component — this helps identify whether a bug is in the component itself or its input preparation

   **For ONNX:** Use ONNX Runtime's `InferenceSession` with intermediate node outputs.

   **For GGUF/transformers:** Hook into the model's forward pass with PyTorch hooks (`register_forward_hook`).

   **For TFLite:** Use `tf.lite.Interpreter` with `get_tensor()` to read intermediate tensor values by index.

   **ONNX activation dumping approach:**
   ```python
   # Key steps:
   # 1. Load model, add all intermediate node outputs to graph
   # 2. Use REALISTIC input (np.random.randn, NOT zeros — zeros miss bugs)
   # 3. Run sess.run(intermediate_names, inputs)
   # 4. Save each as .npy: np.save(f"models/activations/{i:03d}_{name}.npy", arr)
   ```

   See `templates/dump_activations_onnx.py` for a complete working template.

   **For multi-model pipelines:** Dump activations for EACH model separately. Also dump the glue logic outputs (e.g., NMS results, crop coordinates).

3. **Document the activation checkpoint map** in architecture docs: which `.npy` file corresponds to which pipeline stage, with expected shapes.

4. **Verify determinism:** Run the dumper twice and confirm all activations are bitwise identical.

**Success criteria:**
- Every major pipeline stage has a corresponding `.npy` activation file
- Activation shapes documented and consistent with model architecture
- Deterministic across runs
- At least 20+ checkpoints for complex models

---

### Phase 4: Shader Library

**Goal:** Write all WGSL compute shaders needed for this model's forward pass.

**Inputs:** Operation inventory from Phase 1. Model architecture from documentation.

**Steps:**

1. **Map each model operation to a shader.** Prioritize in this order:

   **Essential (almost every model needs these):**
   - `matmul` — tiled 16x16 with workgroup shared memory
   - `add` — element-wise addition (residual connections)
   - Normalization: `layerNorm` or `rmsNorm` or `instanceNorm` (pick based on model)
   - Activation: `gelu` or `relu` or model-specific (Snake, PReLU, etc.)
   - `embeddingLookup` — token/input to vector

   **Add as needed (model-specific):**
   - Attention: `rope`, `kvCacheStore`, `attnScore`, `softmax`, `attnOutput`
   - Convolution: `conv1d`, `conv2d`, `depthwise`, `conv1x1`, `convTranspose1d`
   - Quantized: `linearQ8`, `linearQ4`, `embeddingLookupQ8`
   - Recurrent: `lstm` (bidirectional, one workgroup per direction)
   - Style: `adain` (both channel-first and row-major variants)
   - Reshape: `transpose`, `concat`, `expand`, `pad`, `upsample`
   - I/O: `canvasInput`, `letterboxResize`, `affineCrop`, `argmax`, `topk`
   - Audio: `snake` activation, `istft`, `sinGenerator`
   - Fused: `matmulGelu`, `geluMul`, `fusedNormAdd`, `fusedPerHeadNormRope`

2. **Write `src/shaders.ts`** — all shaders as exported TypeScript string literals.

   **Every shader MUST follow this template:**

   ```wgsl
   // Bind group layout:
   // @group(0) @binding(0) input: array<f32>   [N elements]
   // @group(0) @binding(1) weight: array<f32>  [M x N elements]
   // @group(0) @binding(2) output: array<f32>  [M elements]
   // @group(0) @binding(3) params: Params       {M: u32, N: u32}

   struct Params {
     M: u32,
     N: u32,
   }

   @group(0) @binding(0) var<storage, read> input: array<f32>;
   @group(0) @binding(1) var<storage, read> weight: array<f32>;
   @group(0) @binding(2) var<storage, read_write> output: array<f32>;
   @group(0) @binding(3) var<uniform> params: Params;

   @compute @workgroup_size(256)
   fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
     let idx = gid.x;
     if (idx >= params.M) { return; }  // bounds check

     // ... shader logic here ...

     output[idx] = result;
   }
   ```

   **Key patterns:**
   - Workgroup size 256 (tested on Apple M-series/A-series GPUs; use 64 for broader Intel/AMD iGPU compatibility)
   - ALL dimensions via uniform buffer (never hardcoded)
   - Bounds check at top of main function
   - Document bind group layout in comment header

   **Tiled matmul template** — the workhorse shader. Use 16x16 tiles with `var<workgroup>` shared memory, `workgroupBarrier()` between load and compute phases.

   **Two-pass reduction pattern** (required for softmax, layerNorm, instanceNorm):
   WebGPU has no global synchronization — reductions need two dispatches:
   - **Pass 1:** Each workgroup tree-reduces its chunk via `var<workgroup>` shared memory + `workgroupBarrier()`, writes partial result
   - **Pass 2:** Final reduce + apply (subtract max, exp, divide by sum)
   - For small vectors (hidden_size <= 4096): single-workgroup dispatch with strided loop

3. **Apply numerical stability patterns:**
   - **Clamp tanh/sigmoid inputs** to [-44, 44] — `exp(88)` overflows f32 → NaN propagates through entire inference
   - **Clamp all exp() inputs** — any shader using `exp()` needs clamping
   - **Use vec4 dot products** where possible for 4x fewer iterations

4. **Float16 support** (recommended for vision, optional for others):
   - Write `applyF16Weights()` transformer: converts `array<f32>` weight declarations to `array<f16>`
   - Add f16 validation test at initialization (two-pass compute with realistic buffers)
   - Automatic fallback to f32 when f16 silently fails (iOS Safari)

5. **Shader compilation — always check for errors and use async:**
   ```typescript
   // Use async compilation to avoid blocking the main thread
   const pipeline = await device.createComputePipelineAsync({
     layout: 'auto',
     compute: { module, entryPoint: 'main' }
   });

   // Always check for compilation errors (vary across GPU vendors)
   const info = await module.getCompilationInfo();
   for (const msg of info.messages) {
     if (msg.type === 'error') console.error(`Shader error: ${msg.message} at line ${msg.lineNum}`);
   }
   ```
   A shader that compiles on Metal may fail on Vulkan/D3D12 — always check `getCompilationInfo()`.

6. **Float16 fallback strategy:**
   ```typescript
   // Request f16 if available, gracefully degrade
   const useF16 = adapter.features.has('shader-f16');
   // In shaders, use alias for easy fallback:
   // enable f16;  (only in f16 variant)
   // alias real = f16;  vs  alias real = f32;
   ```
   Generate both f32 and f16 shader variants. Test f16 with a two-pass validation at init — iOS Safari may report `shader-f16` as available but produce incorrect results on complex pipelines.

**Success criteria:**
- Every model operation has a corresponding shader
- Shaders compile via `device.createComputePipelineAsync()` (not sync variant)
- `getCompilationInfo()` checked for all shader modules
- No hardcoded dimensions
- Numerical stability guards on all exp/tanh/sigmoid
- Matmul uses tiled shared-memory variant

**Key lessons learned:**
- The matmul shader dominates performance. Get tiling right early.
- For GGUF: matmul must dequantize on-the-fly using `extractBits()` on packed u32 values
- Snake activation is easy to miss — looks like a residual but has a sin^2 term critical for audio quality
- AdaIN needs both channel-first and row-major variants depending on data layout
- **Group/depthwise convolution:** some models have very high group counts (e.g., `group=1090` in a ConvTranspose1d). Getting the group parameter wrong produces correct-shaped but wrong-valued output. Implement per-group weight slicing in the conv shader.

---

### Phase 4.5: WebGPU Device Limits

**Check these BEFORE writing engine code (Phase 5).** Hard-coding assumptions causes silent failures on different hardware.

```typescript
const adapter = await navigator.gpu.requestAdapter();
const limits = adapter.limits;
// Key limits to check:
// maxStorageBufferBindingSize — typically 128-256MB (can be 1GB+ on desktop)
// maxBufferSize — max single buffer allocation
// maxComputeWorkgroupsPerDimension — 65535 (split dispatches for large tensors)
// maxStorageBuffersPerShaderStage — typically 8 (constrains bind group design)
// maxComputeInvocationsPerWorkgroup — 256 on Apple, sometimes 1024 elsewhere
```

**When tensors exceed limits:**
- **Buffer >maxStorageBufferBindingSize:** Split weight into multiple buffers, dispatch per-chunk
- **Dispatch >65535 workgroups:** Use 2D/3D dispatch grid or process in multiple dispatches
- **>8 storage buffers per shader:** Merge small buffers or split into multiple dispatches

**Memory estimation** (check BEFORE loading):
```
Peak GPU memory = weight_bytes + (2 x largest_activation_layer x 4) + work_buffers
```
For a 1B param Q8_0 model: ~1GB weights + ~50MB activations + ~50MB work = 1.1GB. iPhone has ~4GB shared — leaves headroom but not much.

**Attention-heavy models** (transformers): multiply activation estimate by 3-4x because Q, K, V, attention scores, and residual connections are all live simultaneously during each layer's forward pass.

---

### Phase 5: Engine (Forward Pass)

**Goal:** Write the TypeScript inference engine that orchestrates the full forward pass.

**Inputs:** Shaders from Phase 4. Weight format from Phase 2. Pipeline architecture from Phase 1.

**Steps:**

1. **Write `src/weights.ts`** — the weight loader:
   - For ONNX: custom minimal protobuf parser (not google-protobuf library)
   - For GGUF: the `gguf.ts` from Phase 2 with Range request streaming
   - For TFLite: fetch JSON manifest + binary blob
   - Upload each weight tensor to `GPUBuffer` with `GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC`
   - **Handle buffer allocation failures** — on mobile, `createBuffer` can fail silently or throw when GPU memory is exhausted:
     ```typescript
     function safeCreateBuffer(device: GPUDevice, desc: GPUBufferDescriptor): GPUBuffer | null {
       try {
         const buf = device.createBuffer(desc);
         if (!buf) return null;
         return buf;
       } catch (e) {
         console.error(`Buffer allocation failed (${desc.size} bytes):`, e);
         return null; // caller should try smaller size or lower precision
       }
     }
     ```
     Fallback strategy: try f16 weights (halves memory), reduce max sequence length, or show user a "model too large for this device" message.
   - **CRITICAL: Always include `COPY_SRC` flag** on ALL storage buffers — without it, `copyBufferToBuffer` silently no-ops and debug readbacks return all zeros
   - For tied weights (embedding = LM head): share the same GPUBuffer, not duplicate. Duplicating doubles memory and causes OOM on mobile.
   - For large models (>100MB): Range request streaming — fetch per-layer, upload, free JS buffer. Peak JS memory stays ~50MB instead of model-size.
   - **After all weight uploads:** call `await device.queue.onSubmittedWorkDone()` — this sync point ensures all weight data is committed to GPU before proceeding to pipeline creation.

2. **Write `src/engine.ts`** — the forward pass orchestrator:

   **Write `src/types.ts`** first — define all config interfaces, buffer types, and bind group cache types:
   ```typescript
   // Model config (extracted from GGUF metadata or ONNX graph)
   interface ModelConfig {
     hiddenSize: number;
     numLayers: number;
     numHeads: number;
     // ... model-specific fields
   }

   // Pre-allocated GPU buffers for intermediate values
   interface WorkBuffers {
     hidden: GPUBuffer;
     normed: GPUBuffer;
     residual: GPUBuffer;
     // ... all work buffers
   }

   // Per-layer weight buffers
   interface LayerBuffers {
     attnNormWeight: GPUBuffer;
     qWeight: GPUBuffer;
     // ... all weight buffers for one layer
   }

   // Pre-created bind groups for one layer
   interface LayerBindGroups {
     attnNorm: GPUBindGroup;
     qProj: GPUBindGroup;
     // ... all bind groups for one layer
   }
   ```

   **Initialization sequence** (5 steps, all at load time — zero per-inference allocation):
   1. Request adapter (`powerPreference: 'high-performance'`) + device (request `shader-f16` if available, max buffer limits)
   2. Create ALL shader modules + check `getCompilationInfo()` for errors
   3. Create ALL compute pipelines via `createComputePipelineAsync` (layout: `'auto'`)
   4. Allocate ALL work buffers (always include `STORAGE | COPY_SRC` flags)
   5. Pre-create ALL bind groups (one per dispatch per layer — never recreate per-inference)

   **Forward pass pattern — command encoder batching:**
   - Create ONE `CommandEncoder`, dispatch ALL ops into it, submit ONCE
   - Each dispatch: `beginComputePass()` → `setPipeline` → `setBindGroup` → `dispatchWorkgroups` → `end()`
   - Workgroup counts: 1D = `Math.ceil(N / 256)`, 2D matmul = `[Math.ceil(M/16), Math.ceil(N/16)]`
   - **iOS Safari limit:** flush every ~20-30 dispatches via `device.queue.submit([encoder.finish()])` + new encoder
   - Command encoder batching alone can yield 5-7x speedup

   **Buffer lifecycle:** Use `deferDestroy()` pattern — never destroy buffers that pending command encoders reference. Flush destroys after `queue.submit()` completion.

   **Dynamic shapes** (variable sequence length, phoneme count, etc.):
   - Pre-allocate work buffers to the MAX supported size (e.g., maxSeqLen=2048 for LLMs)
   - Pass actual dimensions via uniform params — shaders bounds-check against actual, not max
   - Bind groups can be created once with max-size buffers — no need to recreate per-inference
   - For attention KV cache: allocate `[maxSeqLen, numHeads, headDim]`, track `currentPos` in uniform

   **Debug support:**
   - `debugCapture` flag enables GPU buffer readbacks at each pipeline stage
   - `readBuffer()` helper: creates staging buffer with `MAP_READ`, copies, maps, returns Float32Array
   - Store in `debugActivations: Map<string, Float32Array>`

3. **Write `src/index.ts`** — the public API:
   - `loadModel(url, options?)` with progress callback
   - Model-specific inference: `generate()` / `synthesize()` / `detect()`
   - Cleanup/destroy method

4. **For multi-model pipelines** (e.g., detector + landmark):
   - Separate `compileModel()` for each model
   - CPU-side glue between models: NMS (non-maximum suppression), ROI computation, coordinate transforms
   - GPU affine crop shader to extract regions for the second model
   - One Euro filter or similar temporal smoothing for video streams

**Success criteria:**
- Engine compiles and runs without GPU errors
- Forward pass produces output of correct shape
- All buffer flags include `COPY_SRC`
- Command encoder batching implemented
- No per-inference bind group or pipeline creation
- Debug capture mode works

**Key lessons learned:**
- Missing `COPY_SRC` is a silent killer — `readBuffer()` returns zeros with no error
- Individual `queue.submit()` per dispatch causes 5-7x slowdown AND crashes iOS Safari
- `mappedAtCreation` for uniform buffers avoids `writeBuffer` overhead
- ONNX `DynamicQuantizeLSTM` stores weights transposed vs standard LSTM — swap `W[gate * IS + j]` to `W[j * H4 + gate]`
- For LLMs: `forwardPassOnly()` during prefill skips unnecessary GPU readbacks between tokens

---

### Phase 6: Activation Matching

**Goal:** Validate correctness by comparing every intermediate WebGPU activation against Phase 3 references. This is where 90% of debugging happens.

**Inputs:** Engine from Phase 5. Reference activations from Phase 3.

**Steps:**

1. **Write `tests/activation-match.spec.ts`** — a Playwright test:

   **Playwright config for WebGPU:**
   ```typescript
   // playwright.config.ts
   export default defineConfig({
     use: {
       browserName: 'chromium',
       launchOptions: {
         args: [
           '--enable-unsafe-webgpu',
           '--enable-features=Vulkan',
           // On macOS, alternatively:
           // '--use-angle=metal',
         ]
       }
     }
   });
   ```

   The test:
   - Loads model with `debugCapture: true`
   - Runs inference on the EXACT same input as reference dumping
   - For each checkpoint: compare with `|a - b| <= atol + rtol * |b|`
   - Report: max absolute error, mean absolute error, pass/fail per checkpoint

2. **Run and iterate.** The first run WILL fail. Debug systematically.

3. **Debugging strategy:**
   - Start from the FIRST failing checkpoint
   - Read back the INPUT to that stage — if input is wrong, bug is upstream
   - Compare shapes first, then values
   - Error magnitude guide: >100x = wrong weights/missing dequant. ~2x = missing sqrt(2). Small systematic = wrong activation or padding

4. **Tighten tolerances** as bugs are fixed:
   - f32: `atol=0.01, rtol=0.01`
   - f16: `atol=0.1, rtol=0.05`
   - Final output: model-dependent (audio ~1.0, landmarks ~0.01)

**Success criteria:**
- ALL activation checkpoints pass within tolerance
- Final output matches reference (correct text, recognizable audio, accurate landmarks)
- No NaN or Inf values anywhere
- Fix bugs in pipeline order — don't fix downstream while upstream is broken

---

### Phase 7: Optimize

**Goal:** Maximize inference speed. Target: competitive with or faster than framework alternatives.

**Inputs:** Correct engine from Phase 6 (all checkpoints passing).

**Steps:**

1. **Benchmark baseline** — profile each stage individually:

   **Profiling template:**
   ```typescript
   async function benchmarkStages(engine: Engine) {
     const times: Record<string, number> = {};
     for (const stage of engine.stages) {
       const start = performance.now();
       stage.dispatch(encoder);
       device.queue.submit([encoder.finish()]);
       await device.queue.onSubmittedWorkDone(); // GPU sync
       times[stage.name] = performance.now() - start;
       encoder = device.createCommandEncoder();
     }
     const sorted = Object.entries(times).sort((a, b) => b[1] - a[1]);
     console.table(sorted); // top entry = bottleneck
   }
   ```

   **GPU-side timing** (more accurate than wall-clock):
   ```typescript
   // Must request feature on BOTH adapter check AND device creation:
   // if (adapter.features.has('timestamp-query'))
   //   device = adapter.requestDevice({ requiredFeatures: ['timestamp-query'] })
   const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
   const resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
   // In compute pass: pass.writeTimestamp(querySet, 0) before, pass.writeTimestamp(querySet, 1) after
   // Then: encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0)
   ```

   **Fused shader pattern** (fusedNormAdd — RMSNorm + residual in one dispatch):
   - Single workgroup of 256 threads, strided loop over hidden dimension
   - Step 1: Each thread accumulates sum-of-squares for its stripe → `var<workgroup>` shared memory
   - Step 2: Tree reduction (`workgroupBarrier` between halving steps) → broadcast RMS to all threads
   - Step 3: Apply `input * rms * weight + residual` in same stripe pattern

2. **Command encoder batching** (5-7x improvement if not done in Phase 5)

3. **Fused shaders** — combine sequential ops on same data:
   - `fusedNormAdd`, `fusedPerHeadNormRope`, `matmulGelu`, `geluMul`

4. **Tiled matmul with shared memory** — 2-3x for attention/FFN

5. **GPU migration** of remaining CPU ops

6. **Buffer lifecycle**: `mappedAtCreation` for uniforms, buffer reuse, drop CPU weight cache after upload

7. **Model-specific:**
   - **LLMs:** KV cache reuse, batched prefill, GPU top-K sampling
   - **TTS:** Per-section HiFi-GAN flushes, cached sin generator weights
   - **Vision:** GPU letterbox/crop, pipelined inference, vec4 dot products, 2-output-channel parallelism

8. **Re-run activation matching** after each optimization.

**Success criteria:**
- At least 2x speedup from Phase 5 baseline
- All checkpoints still pass
- No CPU bottlenecks remaining
- Peak JS memory under 100MB for models under 1GB

---

### Phase 8: Mobile Hardening

**Goal:** Reliable iPhone Safari (iOS 26+) via WebGPU. Every project hits iOS-specific issues.

**Inputs:** Optimized engine from Phase 7.

**Steps:**

1. **Test on real iOS hardware** (or a cloud device service): iPhone Pro, iOS 26+, Safari.

   **Testing workflow:**
   - Deploy demo page to a hosted URL
   - Open Safari DevTools (via remote inspector) to check console for WebGPU errors
   - Run inference 10+ times consecutively to check for memory leaks and thermal throttling
   - Test with the screen locked/unlocked cycle — iOS kills WebGPU contexts on background

   **Backgrounding / thermal:**
   - iOS Safari destroys GPU device when the tab is backgrounded or screen locks
   - The `device.lost` handler (Phase 5) must trigger full re-initialization: re-request adapter+device, re-upload weights, re-create pipelines
   - After 30+ seconds of continuous inference, A-series chips thermal-throttle — expect 20-40% perf drop
   - Never auto-retry on device loss — show user a "tap to restart" button instead

   **Device loss re-initialization pattern:**
   ```typescript
   let device: GPUDevice;

   async function initGPU() {
     const adapter = await navigator.gpu.requestAdapter();
     if (!adapter) throw new Error('No WebGPU adapter');
     device = await adapter.requestDevice({ /* same config */ });

     device.lost.then(async (info) => {
       console.error('GPU device lost:', info.message);
       if (info.reason === 'destroyed') return; // intentional
       // Full re-init: ALL GPU objects become invalid
       await initGPU();         // new device
       await loadWeights(url);  // re-upload all weights
       createPipelines();       // re-create shader modules + pipelines
       createBindGroups();      // re-create all bind groups
       onDeviceRestored?.();    // notify UI
     });
   }
   ```
   - **Never auto-retry inference** after device loss — show "tap to restart" button
   - Device loss is common on iOS: backgrounding, memory pressure, thermal throttling
   - All GPU objects (buffers, pipelines, bind groups) become invalid — must recreate everything

2. **Apply known iOS fixes:**

   **Memory:**
   - Keep large embeddings quantized on GPU (don't dequantize to f32)
   - Drop CPU weight cache after upload
   - Range request streaming for models >100MB
   - Cap context length / max sequence length

   **GPU submission:**
   - Stage-boundary `flushBatchEncoder()` — iOS can't handle 100+ dispatches per submit
   - Per-section flushes within large stages
   - `deferDestroy()` — never destroy a buffer pending command encoders reference

   **f16:** See Phase 4 step 6 for f16 fallback strategy (validation test + alias pattern).

   **Safari compat:** See Phase 2 for `ReadableStream` workaround. Also:
   - Add timeouts on WASM preprocessing with fallbacks
   - Disable autostart on mobile (prevents crash loops)

3. **Performance targets (Apple A18/M4):**
   - <50M params: real-time (2.2ms hand detection)
   - 50-100M params: 1-5x RT (1.3x for TTS)
   - 100M-1B params: usable (34 t/s LLM)
   - >1B: too large for mobile — use smaller variant

**Success criteria:**
- Inference completes without crashing
- No silent f16 failures
- Peak JS memory under 100MB
- Crash-free across 10 consecutive runs

---

### Phase 9: Package and Publish

**Goal:** Build and publish as a zero-dependency npm package with CDN-hosted weights.

**Inputs:** Hardened engine from Phase 8.

**Steps:**

1. **Configure build:** Use `tsup` (ESM, dts, minify). Zero `dependencies` — everything bundled. `devDependencies` only: tsup, typescript, playwright. See `templates/tsup.config.ts` and `templates/package.json` for starter configs.

2. **Host weights:**
   - For npm-published packages: weights auto-served via jsdelivr CDN (`https://cdn.jsdelivr.net/npm/{package}@{version}/models/...`)
   - For larger models: separate hosting with Range request support
   - Default weight URL in code points to CDN; API allows custom URL override

3. **Create demo `index.html`:**
   - Standalone HTML page that imports from the built package
   - Shows model loading progress
   - Interactive inference with the model
   - Mobile-responsive layout
   - Benchmark display (time, tokens/sec or RT factor)

4. **Write README.md:**
   - API documentation with usage examples
   - Performance benchmarks (desktop + mobile)
   - Model variants and sizes
   - Browser requirements (WebGPU, Chrome 113+, Safari iOS 26+)

5. **Build and verify:** `npm run build` must pass before publishing.

6. **Publish:** `npm publish` (or `npm publish --access public` for scoped packages)

**Success criteria:**
- `npm run build` produces clean output with no errors
- Package installs and works from npm: `import { loadModel } from '{package}'`
- Weights load from CDN without CORS issues
- Demo page works on desktop Chrome and mobile Safari
- README has complete API docs and benchmarks

---

## Known Patterns Across All Reference Projects

### Bundle Size Achievement
- Zero runtime dependencies
- Shaders as inline string literals (no .wgsl files to fetch)
- Custom minimal parsers (not google-protobuf, not ggml.js)
- Tree-shakeable ESM exports
- `tsup` for building

### Weight Serving Strategy
- **<50MB:** Single blob, one fetch
- **50-200MB:** Single blob + progress callback
- **>200MB:** Range request streaming (per-layer fetch + upload + free)
- **CDN:** jsdelivr for npm packages, or any static server with Range support

### Development Timeline
Each project typically takes 3-4 days:
- Day 1: Phases 1-3 (inspect, extract, reference)
- Day 2: Phases 4-6 (shaders, engine, activation matching + debugging)
- Day 3: Phases 7-9 (optimize, mobile harden, publish)
