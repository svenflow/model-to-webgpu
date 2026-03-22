/**
 * Public API — model-to-webgpu starter template
 *
 * Replace MODEL_NAME, ModelConfig, and inference method with your model's specifics.
 */

export interface LoadOptions {
  /** URL to the model weights (binary blob or GGUF file) */
  weightsUrl: string;
  /** Progress callback: (loaded, total) => void */
  onProgress?: (loaded: number, total: number) => void;
  /** Force f32 even if f16 is available (for debugging) */
  forceF32?: boolean;
  /** Enable debug activation capture */
  debug?: boolean;
}

export interface Model {
  /** Run inference. Replace with model-specific method signature. */
  infer(input: Float32Array): Promise<Float32Array>;
  /** Free all GPU resources */
  destroy(): void;
  /** Whether the model is using f16 weights */
  readonly usingF16: boolean;
  /** Device info string */
  readonly deviceInfo: string;
}

/**
 * Load the model and return an inference handle.
 *
 * Usage:
 *   const model = await loadModel({ weightsUrl: '/models/weights.bin' });
 *   const output = await model.infer(inputData);
 *   model.destroy();
 */
export async function loadModel(options: LoadOptions): Promise<Model> {
  const { weightsUrl, onProgress, forceF32 = false, debug = false } = options;

  // 1. Request WebGPU adapter + device
  const adapter = await navigator.gpu?.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) throw new Error('WebGPU not supported');

  const useF16 = !forceF32 && adapter.features.has('shader-f16');
  const requiredFeatures: GPUFeatureName[] = [];
  if (useF16) requiredFeatures.push('shader-f16' as GPUFeatureName);

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  const adapterInfo = await adapter.requestAdapterInfo?.() ?? { description: 'unknown' };

  // 2. Handle device loss
  device.lost.then((info) => {
    console.error('GPU device lost:', info.message);
    if (info.reason === 'destroyed') return;
    // Caller should catch this and re-init
  });

  // 3. Load weights
  // TODO: Replace with your weight loading logic (see weights.ts template)
  const weightsResponse = await fetch(weightsUrl);
  if (!weightsResponse.ok) throw new Error(`Failed to fetch weights: ${weightsResponse.status}`);
  const contentLength = Number(weightsResponse.headers.get('content-length') ?? 0);
  const reader = weightsResponse.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, contentLength);
  }

  const weightsData = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    weightsData.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // 4. Create engine (TODO: replace with your Engine class)
  // const engine = new Engine(device, weightsData, { useF16, debug });
  // await engine.compile();

  // 5. Wait for all GPU work to complete before returning
  await device.queue.onSubmittedWorkDone();

  return {
    async infer(input: Float32Array): Promise<Float32Array> {
      // TODO: Replace with your inference logic
      // return engine.forward(input);
      throw new Error('Not implemented — replace with your model inference');
    },

    destroy() {
      // TODO: Destroy all GPU buffers
      // engine.destroy();
      device.destroy();
    },

    get usingF16() {
      return useF16;
    },

    get deviceInfo() {
      return `${(adapterInfo as any).description ?? 'WebGPU'} (f16: ${useF16})`;
    },
  };
}
