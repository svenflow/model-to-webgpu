/**
 * Forward Pass Engine — skeleton with types, init sequence, and dispatch pattern.
 *
 * Adapt this skeleton to your model:
 *   1. Fill in ModelConfig with your model's dimensions
 *   2. Add per-layer weight/bind-group interfaces
 *   3. Implement the forward pass dispatch sequence
 *   4. Wire up the public API (loadModel, generate/infer)
 */

// =============================================================================
// Types
// =============================================================================

/** Model configuration — extracted from GGUF metadata, ONNX graph, or TFLite. */
export interface ModelConfig {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  maxSeqLen: number;
  // Add model-specific fields here
}

/** Pre-allocated GPU buffers for intermediate activations. */
interface WorkBuffers {
  hidden: GPUBuffer;
  normed: GPUBuffer;
  residual: GPUBuffer;
  attnOut: GPUBuffer;
  ffnOut: GPUBuffer;
  // Add more as needed for your model
}

/** Per-layer weight buffers. */
interface LayerWeights {
  normWeight: GPUBuffer;
  normBias: GPUBuffer;
  qWeight: GPUBuffer;
  kWeight: GPUBuffer;
  vWeight: GPUBuffer;
  oWeight: GPUBuffer;
  ffnUpWeight: GPUBuffer;
  ffnDownWeight: GPUBuffer;
  // Add more as needed
}

/** Pre-created bind groups for one layer — never recreated per-inference. */
interface LayerBindGroups {
  norm: GPUBindGroup;
  qProj: GPUBindGroup;
  kProj: GPUBindGroup;
  vProj: GPUBindGroup;
  oProj: GPUBindGroup;
  ffnUp: GPUBindGroup;
  ffnDown: GPUBindGroup;
  // Add more as needed
}

/** All compute pipelines, created once at init. */
interface Pipelines {
  matmul: GPUComputePipeline;
  layerNormStats: GPUComputePipeline;
  layerNormApply: GPUComputePipeline;
  gelu: GPUComputePipeline;
  add: GPUComputePipeline;
  embeddingLookup: GPUComputePipeline;
  softmaxStats: GPUComputePipeline;
  softmaxApply: GPUComputePipeline;
  // Add more as needed
}

/** Engine options. */
export interface EngineOptions {
  /** Enable GPU buffer readbacks at each pipeline stage for debugging. */
  debugCapture?: boolean;
  /** Progress callback during weight loading (0-1). */
  onProgress?: (progress: number) => void;
}

// =============================================================================
// Engine
// =============================================================================

export class Engine {
  private device!: GPUDevice;
  private config!: ModelConfig;
  private pipelines!: Pipelines;
  private workBuffers!: WorkBuffers;
  private layerWeights: LayerWeights[] = [];
  private layerBindGroups: LayerBindGroups[] = [];
  private debugCapture: boolean;
  debugActivations: Map<string, Float32Array> = new Map();

  constructor(options: EngineOptions = {}) {
    this.debugCapture = options.debugCapture ?? false;
  }

  // ===========================================================================
  // Initialization (5 steps — all at load time, zero per-inference allocation)
  // ===========================================================================

  async init(modelUrl: string, options: EngineOptions = {}): Promise<void> {
    // Step 1: Request adapter + device
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("No WebGPU adapter available");

    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has("shader-f16")) {
      requiredFeatures.push("shader-f16");
    }
    if (adapter.features.has("timestamp-query")) {
      requiredFeatures.push("timestamp-query");
    }

    this.device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });

    // Handle device loss (critical for iOS Safari)
    this.device.lost.then(async (info) => {
      console.error("GPU device lost:", info.message);
      if (info.reason === "destroyed") return;
      // Full re-init required — all GPU objects are invalid
      await this.init(modelUrl, options);
    });

    // Step 2: Create shader modules + check compilation
    await this.createShaderModules();

    // Step 3: Create compute pipelines (async)
    await this.createPipelines();

    // Step 4: Load weights and allocate work buffers
    await this.loadWeights(modelUrl, options.onProgress);

    // Sync point: ensure all weight data is committed to GPU
    await this.device.queue.onSubmittedWorkDone();

    // Step 5: Pre-create all bind groups
    this.createBindGroups();
  }

  private async createShaderModules(): Promise<void> {
    // TODO: Import your shaders from shaders.ts and create modules
    // Example:
    //
    // const matmulModule = this.device.createShaderModule({ code: matmulShader });
    // const info = await matmulModule.getCompilationInfo();
    // for (const msg of info.messages) {
    //   if (msg.type === 'error') {
    //     throw new Error(`Shader compilation error: ${msg.message} at line ${msg.lineNum}`);
    //   }
    // }
  }

  private async createPipelines(): Promise<void> {
    // TODO: Create all pipelines with createComputePipelineAsync
    // Example:
    //
    // this.pipelines.matmul = await this.device.createComputePipelineAsync({
    //   layout: 'auto',
    //   compute: { module: matmulModule, entryPoint: 'main' },
    // });
  }

  private async loadWeights(
    _modelUrl: string,
    _onProgress?: (progress: number) => void,
  ): Promise<void> {
    // TODO: Fetch and upload weights
    // For each weight tensor:
    //   1. Fetch data (Range requests for large models)
    //   2. Apply format transformations (dequantize, transpose, etc.)
    //   3. Upload to GPUBuffer with STORAGE | COPY_SRC flags
    //   4. Free JS-side buffer
    //
    // Example buffer creation:
    //
    // const buffer = this.device.createBuffer({
    //   size: data.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    //   mappedAtCreation: true,
    // });
    // new Float32Array(buffer.getMappedRange()).set(data);
    // buffer.unmap();

    // Allocate work buffers (max size for dynamic shapes)
    this.allocateWorkBuffers();
  }

  private allocateWorkBuffers(): void {
    // TODO: Allocate all intermediate buffers at max supported size
    // Example:
    //
    // const maxHiddenBytes = this.config.maxSeqLen * this.config.hiddenSize * 4;
    // this.workBuffers.hidden = this.device.createBuffer({
    //   size: maxHiddenBytes,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    // });
  }

  private createBindGroups(): void {
    // TODO: Pre-create ALL bind groups for ALL layers
    // One bind group per dispatch per layer — never recreate per-inference
    //
    // Example:
    //
    // for (let l = 0; l < this.config.numLayers; l++) {
    //   const bg: LayerBindGroups = {
    //     norm: this.device.createBindGroup({
    //       layout: this.pipelines.layerNormStats.getBindGroupLayout(0),
    //       entries: [
    //         { binding: 0, resource: { buffer: this.workBuffers.hidden } },
    //         { binding: 1, resource: { buffer: statsBuffer } },
    //         { binding: 2, resource: { buffer: paramsBuffer } },
    //       ],
    //     }),
    //     // ... more bind groups
    //   };
    //   this.layerBindGroups.push(bg);
    // }
  }

  // ===========================================================================
  // Forward Pass — command encoder batching
  // ===========================================================================

  async forward(_input: unknown): Promise<Float32Array> {
    let encoder = this.device.createCommandEncoder();
    let dispatchCount = 0;

    // Helper: dispatch a compute pass
    const dispatch = (
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      workgroups: [number, number?, number?],
      label?: string,
    ) => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...workgroups);
      pass.end();
      dispatchCount++;

      // iOS Safari: flush every ~25 dispatches
      if (dispatchCount >= 25) {
        this.device.queue.submit([encoder.finish()]);
        encoder = this.device.createCommandEncoder();
        dispatchCount = 0;
      }

      // Debug capture
      if (this.debugCapture && label) {
        // Queue a readback after this dispatch
        // (actual readback happens after submit)
      }
    };

    // TODO: Implement forward pass
    // Example structure:
    //
    // // Embedding lookup
    // dispatch(this.pipelines.embeddingLookup, embedBG, [Math.ceil(dim / 256)], 'embedding');
    //
    // // Transformer layers
    // for (let l = 0; l < this.config.numLayers; l++) {
    //   // LayerNorm
    //   dispatch(this.pipelines.layerNormStats, this.layerBindGroups[l].norm, [rows]);
    //   dispatch(this.pipelines.layerNormApply, normApplyBG, [Math.ceil(total / 256)]);
    //
    //   // Attention
    //   dispatch(this.pipelines.matmul, this.layerBindGroups[l].qProj, [M16, N16], `layer${l}_qkv`);
    //   // ... softmax, attention output ...
    //
    //   // Residual add
    //   dispatch(this.pipelines.add, addBG, [Math.ceil(total / 256)]);
    //
    //   // FFN
    //   dispatch(this.pipelines.matmul, this.layerBindGroups[l].ffnUp, [M16, N16]);
    //   dispatch(this.pipelines.gelu, geluBG, [Math.ceil(intermediate / 256)]);
    //   dispatch(this.pipelines.matmul, this.layerBindGroups[l].ffnDown, [M16, N16]);
    //
    //   // Residual add
    //   dispatch(this.pipelines.add, addBG2, [Math.ceil(total / 256)]);
    // }

    // Final submit
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // Read output
    return this.readBuffer(this.workBuffers.hidden, 0 /* byte size */);
  }

  // ===========================================================================
  // Debug Utilities
  // ===========================================================================

  /**
   * Read a GPU buffer back to CPU.
   * Creates a staging buffer with MAP_READ, copies, maps, returns Float32Array.
   *
   * IMPORTANT: Source buffer MUST have COPY_SRC flag or this returns all zeros.
   */
  private async readBuffer(source: GPUBuffer, byteSize: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    return data;
  }

  /** Clean up all GPU resources. */
  destroy(): void {
    // Destroy all buffers
    if (this.workBuffers) {
      for (const buf of Object.values(this.workBuffers)) {
        (buf as GPUBuffer).destroy();
      }
    }
    for (const layer of this.layerWeights) {
      for (const buf of Object.values(layer)) {
        (buf as GPUBuffer).destroy();
      }
    }
    this.device.destroy();
  }
}
