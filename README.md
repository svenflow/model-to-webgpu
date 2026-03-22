# model-to-webgpu 🔨

Convert any ML model into a standalone browser project with hand-written WGSL compute shaders. No runtime framework, no ONNX Runtime, no transformers.js. The output is a zero-dependency TypeScript package that runs inference entirely via WebGPU compute dispatches.

## Why Hand-Written Shaders?

| | Framework (ONNX Runtime Web) | model-to-webgpu |
|---|---|---|
| **Bundle size** | 2-5MB runtime | 8-12KB gzipped |
| **Dependencies** | onnxruntime-web, transformers.js | Zero |
| **Shaders** | Auto-generated from graph | Hand-written WGSL |
| **Optimization** | Generic | Model-specific |
| **Debuggability** | Black box | Every shader readable |

## What It Produces

Given a model file (ONNX, GGUF, or TFLite), this methodology guides you through generating a complete standalone repo:

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

The output is NOT a generic runtime. Every generated file is model-specific, readable, and modifiable. The shaders are hand-written WGSL — not auto-generated from a graph compiler.

## Reference Implementations

These completed projects demonstrate the methodology in practice:

| Project | Model Type | Source Format | Shaders | Bundle | Perf (M4 Pro) |
|---------|-----------|---------------|---------|--------|---------------|
| [webgpu-gemma](https://github.com/nicholasgasior/webgpu-gemma) | LLM (Gemma 3 1B) | GGUF Q8_0 | 18 | 12KB gz | 59.8 tok/s |
| [kitten-tts-webgpu](https://github.com/svenflow/kitten-tts-webgpu) | TTS (Kitten TTS 80M) | ONNX (quantized) | 29 | 753KB gz | 3.3x RT |
| [micro-handpose](https://github.com/svenflow/micro-handpose) | Vision (Hand landmarks) | TFLite | 28 | 8KB gz | 2.2ms |

## When to Use This

✅ **Use model-to-webgpu when:**
- You want the smallest possible bundle for browser ML
- You need zero runtime dependencies
- You want full control over every GPU dispatch
- You're targeting mobile Safari (iOS 26+)
- Model is <1B parameters

❌ **Use ONNX Runtime Web / Transformers.js instead when:**
- Prototyping speed matters more than bundle size
- Model uses only standard ops
- Bundle size is irrelevant (internal tools)
- Model is >1B parameters
- You need broad hardware compatibility beyond Apple GPUs

## The 9-Phase Methodology

Each phase has clear inputs, outputs, and success criteria. The full methodology is documented in [`METHODOLOGY.md`](./METHODOLOGY.md).

| Phase | Name | Goal | Time |
|-------|------|------|------|
| 1 | [Inspect](#phase-1-inspect) | Understand model graph, ops, weights, quantization | 2-3h |
| 2 | [Extract](#phase-2-extract-weights--preprocessing) | Extract weights, build preprocessing pipeline | 3-4h |
| 3 | [Reference](#phase-3-reference-activations) | Generate ground-truth activations | 1-2h |
| 4 | [Shaders](#phase-4-shader-library) | Write all WGSL compute shaders | 4-6h |
| 5 | [Engine](#phase-5-engine) | Build forward pass orchestrator | 4-6h |
| 6 | [Validate](#phase-6-activation-matching) | Match WebGPU output to reference | 4-8h |
| 7 | [Optimize](#phase-7-optimize) | Maximize inference speed | 2-4h |
| 8 | [Mobile](#phase-8-mobile-hardening) | iPhone Safari hardening | 2-3h |
| 9 | [Publish](#phase-9-package-and-publish) | npm package + CDN weights | 1-2h |

**Typical total: 3-4 days** from model file to published npm package.

## Quick Reference

### Shader Patterns

**Tiled matmul** — 16x16 tiles with workgroup shared memory:
```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  var<workgroup> tileA: array<f32, 256>;
  var<workgroup> tileB: array<f32, 256>;
  // Load tile → barrier → accumulate → barrier → next tile
}
```

**Two-pass reduction** (softmax, layerNorm):
```
Pass 1: Each workgroup reduces its chunk → partial results
Pass 2: Final reduce + apply (subtract max, exp, divide by sum)
```

**Fused ops** — combine sequential dispatches:
```
fusedNormAdd: RMSNorm + residual in one dispatch
fusedPerHeadNormRope: per-head norm + rotary embedding
matmulGelu: matmul + GELU activation
```

### Numerical Stability

```wgsl
// ALWAYS clamp exp/tanh/sigmoid inputs
let safe_x = clamp(x, -88.0, 88.0);  // exp() overflow guard
let safe_t = clamp(x, -44.0, 44.0);  // tanh/sigmoid guard
```

### WebGPU Device Limits

```typescript
// Check before writing engine code
const limits = adapter.limits;
// maxStorageBufferBindingSize — typically 128-256MB
// maxBufferSize — max single buffer allocation
// maxComputeWorkgroupsPerDimension — 65535
// maxStorageBuffersPerShaderStage — typically 8
// maxComputeInvocationsPerWorkgroup — 256 on Apple
```

### iOS Safari Gotchas

- **Flush every ~20-30 dispatches** — iOS can't handle 100+ per submit
- **f16 may silently fail** — validate with a two-pass compute test at init
- **Device loss on background** — implement full re-initialization handler
- **`ReadableStream[Symbol.asyncIterator]` unsupported** — use manual `getReader()` loop

## Known Bug Database

See [`KNOWN_BUGS.md`](./KNOWN_BUGS.md) for 14 documented failure modes with symptom → cause → fix patterns.

Quick triage by error magnitude:
- **>100x off** → wrong weights or missing dequantization
- **~2x off** → missing `sqrt(2)` residual scaling or `sqrt(hidden_size)` embedding scaling
- **Small systematic** → wrong activation function or padding asymmetry
- **All NaN** → `exp()` overflow — clamp inputs to [-88, 88]
- **All zeros in debug readback** → missing `COPY_SRC` buffer flag

## Templates

The [`templates/`](./templates/) directory contains starter files for common patterns:

- `inspect_model.py` — ONNX/GGUF/TFLite model inspector
- `dump_activations.py` — reference activation dumper
- `shaders.ts` — shader string literal with standard patterns
- `engine.ts` — forward pass orchestrator skeleton
- `activation-match.spec.ts` — Playwright test for WebGPU validation
- `playwright.config.ts` — WebGPU-enabled Playwright config
- `tsup.config.ts` — zero-dependency ESM build config
- `package.json` — starter package with zero dependencies

## License

MIT
