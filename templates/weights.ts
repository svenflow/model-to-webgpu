/**
 * Weight loader — model-to-webgpu starter template
 *
 * Handles: binary blob fetch, JSON manifest parsing, GPU buffer upload.
 * For ONNX/TFLite: use extract_weights.py to produce blob + manifest, then load here.
 * For GGUF: replace with a runtime GGUF parser (see METHODOLOGY.md Phase 2).
 */

export interface WeightManifest {
  /** Total size of the binary blob in bytes */
  totalBytes: number;
  /** Individual tensor entries */
  tensors: TensorEntry[];
}

export interface TensorEntry {
  /** Unique key for this tensor (e.g., "encoder.layer0.attn.q_weight") */
  key: string;
  /** Tensor shape */
  shape: number[];
  /** Data type in the blob */
  dtype: 'float32' | 'float16' | 'int8' | 'uint8';
  /** Byte offset into the binary blob */
  offset: number;
  /** Byte length of this tensor */
  length: number;
}

export interface WeightBuffers {
  /** Map from tensor key to GPU buffer */
  buffers: Map<string, GPUBuffer>;
  /** Total GPU memory used in bytes */
  totalGpuBytes: number;
}

/**
 * Load weights from a binary blob + JSON manifest.
 *
 * Usage:
 *   const manifest = await (await fetch('/models/manifest.json')).json();
 *   const weights = await loadWeights(device, '/models/weights.bin', manifest, onProgress);
 */
export async function loadWeights(
  device: GPUDevice,
  blobUrl: string,
  manifest: WeightManifest,
  onProgress?: (loaded: number, total: number) => void,
): Promise<WeightBuffers> {
  // Fetch the binary blob with progress tracking
  const response = await fetch(blobUrl);
  if (!response.ok) throw new Error(`Weight fetch failed: ${response.status}`);

  const reader = response.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  // Manual reader loop (Safari doesn't support ReadableStream[Symbol.asyncIterator])
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, manifest.totalBytes);
  }

  // Concatenate chunks
  const blob = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    blob.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // Upload each tensor to a GPU buffer
  const buffers = new Map<string, GPUBuffer>();
  let totalGpuBytes = 0;

  for (const tensor of manifest.tensors) {
    const data = blob.slice(tensor.offset, tensor.offset + tensor.length);

    // Convert to f32 if needed (dequantization should be done in extract_weights.py)
    const f32Data = tensorToFloat32(data, tensor.dtype);
    const byteLength = f32Data.byteLength;

    // CRITICAL: Always include COPY_SRC — without it, debug readbacks return all zeros
    const buffer = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(f32Data);
    buffer.unmap();

    buffers.set(tensor.key, buffer);
    totalGpuBytes += byteLength;
  }

  // Sync point: ensure all weight data is committed to GPU
  await device.queue.onSubmittedWorkDone();

  return { buffers, totalGpuBytes };
}

/**
 * Convert raw tensor bytes to Float32Array.
 * Dequantization (INT8/UINT8 with scale+zero_point) should happen
 * in extract_weights.py, not here — keeps GPU code simple.
 */
function tensorToFloat32(data: Uint8Array, dtype: string): Float32Array {
  switch (dtype) {
    case 'float32':
      return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);

    case 'float16': {
      // f16 → f32 conversion
      const f16 = new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2);
      const f32 = new Float32Array(f16.length);
      for (let i = 0; i < f16.length; i++) {
        f32[i] = float16ToFloat32(f16[i]);
      }
      return f32;
    }

    case 'int8': {
      const i8 = new Int8Array(data.buffer, data.byteOffset, data.byteLength);
      const f32 = new Float32Array(i8.length);
      for (let i = 0; i < i8.length; i++) f32[i] = i8[i];
      return f32;
    }

    case 'uint8': {
      const f32 = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) f32[i] = data[i];
      return f32;
    }

    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

/** Convert a float16 (stored as uint16) to float32. */
function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;

  if (exponent === 0) {
    // Subnormal or zero
    if (mantissa === 0) return sign ? -0 : 0;
    // Subnormal: value = (-1)^sign * 2^(-14) * (mantissa / 1024)
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  }
  if (exponent === 31) {
    // Infinity or NaN
    return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  // Normal: value = (-1)^sign * 2^(exponent - 15) * (1 + mantissa / 1024)
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

/**
 * Load weights via Range requests for large models (>100MB).
 * Fetches and uploads one layer at a time to keep peak JS memory low (~50MB).
 */
export async function loadWeightsStreaming(
  device: GPUDevice,
  blobUrl: string,
  manifest: WeightManifest,
  onProgress?: (loaded: number, total: number) => void,
): Promise<WeightBuffers> {
  const buffers = new Map<string, GPUBuffer>();
  let totalGpuBytes = 0;
  let loaded = 0;

  for (const tensor of manifest.tensors) {
    // Range request for just this tensor
    const response = await fetch(blobUrl, {
      headers: { Range: `bytes=${tensor.offset}-${tensor.offset + tensor.length - 1}` },
    });

    const data = new Uint8Array(await response.arrayBuffer());
    const f32Data = tensorToFloat32(data, tensor.dtype);

    const buffer = device.createBuffer({
      size: f32Data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(f32Data);
    buffer.unmap();

    buffers.set(tensor.key, buffer);
    totalGpuBytes += f32Data.byteLength;
    loaded += tensor.length;
    onProgress?.(loaded, manifest.totalBytes);
  }

  await device.queue.onSubmittedWorkDone();

  return { buffers, totalGpuBytes };
}
