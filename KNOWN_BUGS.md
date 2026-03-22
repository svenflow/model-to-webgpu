# model-to-webgpu — Known Bug Database

Reference for Phase 6 activation matching debugging.

| Bug Pattern | Symptom | Root Cause | Fix |
|-------------|---------|------------|-----|
| Wrong weight layout | Early layer diverges by large factor | Weight tensor indexed with wrong stride/dimension order. ONNX DynamicQuantizeLSTM transposes W. TFLite uses NHWC. | Check source format's weight layout docs. Swap indexing. |
| Missing dequantization | All outputs ~100x too large/small | INT8/UINT8 weights uploaded without `(val - zero_point) * scale` | Apply dequant formula during weight extraction |
| Varint parsing error | Specific layers have wrong bias/zero_point | ONNX protobuf packed int32 fields use varint encoding, not raw 4-byte values | Parse varints inside packed repeated fields |
| Missing normalization factor | Error grows multiplicatively through layers | Residual connections missing `/ sqrt(2)`, or embedding missing `* sqrt(hidden_size)` | Check model source for residual scaling conventions |
| Wrong activation function | Shape correct, values systematically wrong | Using ReLU where model uses Snake, or LeakyReLU where model uses PReLU | Trace ONNX graph carefully — names can mislead |
| NaN propagation | All-NaN output after a few layers | `exp()` overflow in GELU/sigmoid/tanh when input >88 | Clamp: tanh/sigmoid to [-44,44], exp to [-88,88] |
| Buffer flag missing | Debug readbacks return all zeros | GPUBuffer without `COPY_SRC` (see Phase 5) | Add `COPY_SRC` to ALL storage buffers |
| Wrong tensor input | Correct shape, wrong values | Feeding wrong intermediate tensor to a component | Trace ONNX graph to confirm exact input tensor |
| Padding asymmetry | Small consistent error in conv layers | TFLite stride-2 SAME padding is asymmetric | Compute asymmetric padding per TFLite's formula |
| Style vector split | Two components produce swapped outputs | Split index wrong (e.g., `[0:128]` vs `[128:256]` swapped) | Check source model's split dimension |
| BN param handling | Conv outputs systematically biased | Using 4 BN params when TFLite fused them into single bias | Check for FusedBatchNormV3 — use fused bias only |
| f16 silent failure | All outputs zero on iOS Safari | `shader-f16` available but complex pipelines produce zeros | Add f16 validation self-test, fallback to f32 |
| Weight key collision | One layer always wrong | Two tensors share same key, Map overwrite | Disambiguate with shape suffix |
| Group conv wrong | Correct shape, wrong channel values | Group/depthwise conv parameter wrong or not implemented | Check group count, implement per-group weight slicing |
| Tied weight duplication | OOM on mobile, 2x expected memory | Embedding and LM head using separate copies of same weights | Share GPUBuffer, don't upload twice |
| workgroupBarrier in branch | Random wrong values, non-deterministic | `workgroupBarrier()` inside divergent `if` block — undefined behavior per WGSL spec | Move ALL barriers outside conditionals, ensure all invocations reach them |
| u32 index overflow | Wrong values for large tensors only | `row * N + col` overflows u32 when tensor has >4B elements | Cast to i32 arithmetic or split into multiple dispatches |

## Error Magnitude Guide

Quick triage based on how wrong the values are:
- **>100x off** = wrong weights or missing dequantization
- **~2x off** = missing `sqrt(2)` residual scaling or `sqrt(hidden_size)` embedding scaling
- **Small systematic** = wrong activation function or padding asymmetry
- **All NaN** = `exp()` overflow — clamp inputs to [-88, 88]
- **All zeros in debug readback** = missing `COPY_SRC` buffer flag
