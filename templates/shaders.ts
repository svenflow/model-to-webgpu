/**
 * WGSL Compute Shaders — starter file with standard templates.
 *
 * Every shader follows the same pattern:
 *   - Bind group layout documented in comment header
 *   - All dimensions via uniform buffer (never hardcoded)
 *   - Bounds check at top of main function
 *   - Workgroup size 256 (Apple M/A-series; use 64 for Intel/AMD iGPU)
 *
 * Adapt these templates to your model's specific tensor shapes and data flow.
 */

// =============================================================================
// Tiled Matrix Multiplication (16x16 tiles with shared memory)
// =============================================================================

/**
 * Tiled matmul: output[M, N] = input[M, K] * weight[K, N]
 *
 * Uses 16x16 workgroup tiles with shared memory for coalesced access.
 * This is the workhorse shader — dominates performance in most models.
 */
export const matmulShader = /* wgsl */ `
// Bind group layout:
// @group(0) @binding(0) input: array<f32>   [M * K elements, row-major]
// @group(0) @binding(1) weight: array<f32>  [K * N elements, row-major]
// @group(0) @binding(2) output: array<f32>  [M * N elements, row-major]
// @group(0) @binding(3) params: Params       {M, N, K}

struct Params {
  M: u32,
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>; // TILE * TILE
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let lr = lid.y;
  let lc = lid.x;

  var sum: f32 = 0.0;
  let numTiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // Load tile from input (M x K)
    let aRow = row;
    let aCol = t * TILE + lc;
    if (aRow < params.M && aCol < params.K) {
      tileA[lr * TILE + lc] = input[aRow * params.K + aCol];
    } else {
      tileA[lr * TILE + lc] = 0.0;
    }

    // Load tile from weight (K x N)
    let bRow = t * TILE + lr;
    let bCol = col;
    if (bRow < params.K && bCol < params.N) {
      tileB[lr * TILE + lc] = weight[bRow * params.N + bCol];
    } else {
      tileB[lr * TILE + lc] = 0.0;
    }

    workgroupBarrier();

    // Accumulate dot product for this tile
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      sum = sum + tileA[lr * TILE + k] * tileB[k * TILE + lc];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    output[row * params.N + col] = sum;
  }
}
`;

// =============================================================================
// Element-wise Addition (residual connections)
// =============================================================================

/**
 * Element-wise add: output[i] = a[i] + b[i]
 */
export const addShader = /* wgsl */ `
// Bind group layout:
// @group(0) @binding(0) a: array<f32>      [N elements]
// @group(0) @binding(1) b: array<f32>      [N elements]
// @group(0) @binding(2) output: array<f32> [N elements]
// @group(0) @binding(3) params: Params      {N}

struct Params {
  N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = a[idx] + b[idx];
}
`;

// =============================================================================
// Layer Normalization (two-pass: compute stats, then normalize)
// =============================================================================

/**
 * LayerNorm Pass 1: Compute mean and variance per row.
 * output[row * 2] = mean, output[row * 2 + 1] = variance
 *
 * For hidden_size <= 4096, use single-workgroup strided reduction.
 */
export const layerNormStatsShader = /* wgsl */ `
// Bind group layout:
// @group(0) @binding(0) input: array<f32>  [rows * cols elements]
// @group(0) @binding(1) stats: array<f32>  [rows * 2 elements: mean, var per row]
// @group(0) @binding(2) params: Params      {rows, cols}

struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f32, 512>; // 256 for sum, 256 for sum_sq

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let tid = lid.x;

  // Strided accumulation
  var sum: f32 = 0.0;
  var sum_sq: f32 = 0.0;
  for (var i = tid; i < params.cols; i = i + 256u) {
    let val = input[row * params.cols + i];
    sum = sum + val;
    sum_sq = sum_sq + val * val;
  }

  shared[tid] = sum;
  shared[tid + 256u] = sum_sq;
  workgroupBarrier();

  // Tree reduction
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared[tid] = shared[tid] + shared[tid + s];
      shared[tid + 256u] = shared[tid + 256u] + shared[tid + 256u + s];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let mean = shared[0] / f32(params.cols);
    let variance = shared[256u] / f32(params.cols) - mean * mean;
    stats[row * 2u] = mean;
    stats[row * 2u + 1u] = variance;
  }
}
`;

/**
 * LayerNorm Pass 2: Apply normalization.
 * output[i] = (input[i] - mean) / sqrt(variance + eps) * weight + bias
 */
export const layerNormApplyShader = /* wgsl */ `
// Bind group layout:
// @group(0) @binding(0) input: array<f32>   [rows * cols]
// @group(0) @binding(1) stats: array<f32>   [rows * 2: mean, var]
// @group(0) @binding(2) weight: array<f32>  [cols]
// @group(0) @binding(3) bias: array<f32>    [cols]
// @group(0) @binding(4) output: array<f32>  [rows * cols]
// @group(0) @binding(5) params: Params       {rows, cols}

struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const EPS: f32 = 1e-5;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.rows * params.cols;
  if (idx >= total) { return; }

  let row = idx / params.cols;
  let col = idx % params.cols;
  let mean = stats[row * 2u];
  let variance = stats[row * 2u + 1u];
  let inv_std = 1.0 / sqrt(variance + EPS);

  output[idx] = (input[idx] - mean) * inv_std * weight[col] + bias[col];
}
`;

// =============================================================================
// GELU Activation
// =============================================================================

/**
 * GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Clamps tanh input to [-44, 44] to prevent f32 overflow.
 */
export const geluShader = /* wgsl */ `
// Bind group layout:
// @group(0) @binding(0) input: array<f32>   [N elements]
// @group(0) @binding(1) output: array<f32>  [N elements]
// @group(0) @binding(2) params: Params       {N}

struct Params {
  N: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }

  let x = input[idx];
  let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
  // Clamp to prevent exp() overflow in tanh
  let clamped = clamp(inner, -44.0, 44.0);
  output[idx] = 0.5 * x * (1.0 + tanh(clamped));
}
`;

// =============================================================================
// ReLU Activation
// =============================================================================

/**
 * ReLU: max(0, x)
 */
export const reluShader = /* wgsl */ `
struct Params {
  N: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = max(0.0, input[idx]);
}
`;

// =============================================================================
// Embedding Lookup
// =============================================================================

/**
 * Embedding lookup: output[i] = embedding_table[token_id * dim + i]
 * Copies one row from the embedding table based on the token index.
 */
export const embeddingLookupShader = /* wgsl */ `
// Bind group layout:
// @group(0) @binding(0) table: array<f32>    [vocab_size * dim]
// @group(0) @binding(1) output: array<f32>   [dim]
// @group(0) @binding(2) params: Params        {token_id, dim}

struct Params {
  token_id: u32,
  dim: u32,
}

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.dim) { return; }
  output[idx] = table[params.token_id * params.dim + idx];
}
`;

// =============================================================================
// Softmax (two-pass: max + sum, then apply)
// =============================================================================

/**
 * Softmax Pass 1: Compute max and sum(exp(x - max)) per row.
 * stats[row * 2] = max, stats[row * 2 + 1] = sum_exp
 */
export const softmaxStatsShader = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let tid = lid.x;

  // Find max (strided)
  var local_max: f32 = -1e30;
  for (var i = tid; i < params.cols; i = i + 256u) {
    local_max = max(local_max, input[row * params.cols + i]);
  }
  shared_max[tid] = local_max;
  workgroupBarrier();

  // Reduce max
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
    }
    workgroupBarrier();
  }
  let row_max = shared_max[0];

  // Sum exp(x - max) (strided)
  var local_sum: f32 = 0.0;
  for (var i = tid; i < params.cols; i = i + 256u) {
    let val = input[row * params.cols + i] - row_max;
    local_sum = local_sum + exp(clamp(val, -88.0, 88.0));
  }
  shared_sum[tid] = local_sum;
  workgroupBarrier();

  // Reduce sum
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    stats[row * 2u] = row_max;
    stats[row * 2u + 1u] = shared_sum[0];
  }
}
`;

/**
 * Softmax Pass 2: Apply exp(x - max) / sum.
 */
export const softmaxApplyShader = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.rows * params.cols;
  if (idx >= total) { return; }

  let row = idx / params.cols;
  let row_max = stats[row * 2u];
  let row_sum = stats[row * 2u + 1u];

  let val = input[idx] - row_max;
  output[idx] = exp(clamp(val, -88.0, 88.0)) / row_sum;
}
`;
