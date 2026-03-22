#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["onnx", "onnxruntime", "numpy"]
# ///
"""ONNX Reference Activation Dumper — Phase 3 template.

Runs inference through an ONNX model and saves every intermediate tensor
as a .npy file. These serve as ground-truth "oracle" activations for
validating the WebGPU implementation (Phase 6 activation matching).

The script:
  1. Loads the ONNX model
  2. Adds all intermediate node outputs to the graph
  3. Runs inference with realistic random input (NOT zeros — zeros miss bugs)
  4. Saves each intermediate activation as models/activations/NNN_name.npy
  5. Prints a checkpoint map with shapes for documentation

Usage:
    ./dump_activations_onnx.py model.onnx
    ./dump_activations_onnx.py model.onnx --output-dir ./activations
    ./dump_activations_onnx.py model.onnx --seed 42
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def get_input_shapes(graph: onnx.GraphProto) -> dict[str, tuple]:
    """Extract input tensor shapes from the graph, replacing dynamic dims with defaults."""
    shapes = {}
    init_names = {init.name for init in graph.initializer}

    for inp in graph.input:
        if inp.name in init_names:
            continue
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                # Dynamic dimension — use sensible default
                dim_name = d.dim_param or "?"
                if "batch" in dim_name.lower():
                    dims.append(1)
                elif "seq" in dim_name.lower() or "length" in dim_name.lower():
                    dims.append(16)
                else:
                    dims.append(8)
        shapes[inp.name] = tuple(dims)
    return shapes


def get_numpy_dtype(elem_type: int) -> np.dtype:
    """Map ONNX element type to numpy dtype."""
    type_map = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.BOOL: np.bool_,
    }
    return np.dtype(type_map.get(elem_type, np.float32))


def dump_activations(
    model_path: str,
    output_dir: str = "models/activations",
    seed: int = 42,
) -> None:
    # Load model
    model = onnx.load(model_path)
    graph = model.graph

    # Collect all intermediate output names (skip graph inputs and initializers)
    init_names = {init.name for init in graph.initializer}
    input_names = {inp.name for inp in graph.input if inp.name not in init_names}
    output_names = {out.name for out in graph.output}

    # Add all intermediate node outputs to graph outputs so ONNX Runtime exposes them
    intermediate_outputs = []
    seen = set()
    for node in graph.node:
        for out_name in node.output:
            if out_name and out_name not in seen and out_name not in output_names:
                intermediate_outputs.append(out_name)
                seen.add(out_name)
                # Add as graph output with unknown shape
                graph.output.append(
                    onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None)
                )

    all_output_names = intermediate_outputs + [out.name for out in graph.output if out.name not in seen]

    # Create session
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # suppress warnings
    session = ort.InferenceSession(model.SerializeToString(), sess_options)

    # Generate realistic random input (NOT zeros — zeros hide bugs like missing dequantization)
    rng = np.random.RandomState(seed)
    input_shapes = get_input_shapes(graph)
    feeds = {}

    print("## Input Tensors")
    for inp in graph.input:
        if inp.name in init_names:
            continue
        shape = input_shapes.get(inp.name, (1,))
        dtype = get_numpy_dtype(inp.type.tensor_type.elem_type)

        if np.issubdtype(dtype, np.floating):
            data = rng.randn(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            data = rng.randint(0, 100, size=shape).astype(dtype)
        elif dtype == np.bool_:
            data = rng.randint(0, 2, size=shape).astype(np.bool_)
        else:
            data = rng.randn(*shape).astype(np.float32)

        feeds[inp.name] = data
        print(f"  {inp.name}: {list(shape)} {dtype} (seed={seed})")
    print()

    # Run inference with all outputs
    print("Running inference...")
    try:
        results = session.run(None, feeds)
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        print("Try adjusting input shapes or checking model compatibility.")
        sys.exit(1)

    # Get output names in order
    session_outputs = [o.name for o in session.get_outputs()]

    # Save activations
    os.makedirs(output_dir, exist_ok=True)

    # Also save inputs
    for name, data in feeds.items():
        save_path = os.path.join(output_dir, f"000_input_{name.replace('/', '_')}.npy")
        np.save(save_path, data)

    print(f"\n## Activation Checkpoints → {output_dir}/")
    print(f"  [000] inputs: {', '.join(f'{k}: {list(v.shape)}' for k, v in feeds.items())}")

    saved_count = 0
    for i, (name, arr) in enumerate(zip(session_outputs, results)):
        if arr is None:
            continue
        arr = np.array(arr)
        safe_name = name.replace("/", "_").replace(":", "_")
        filename = f"{i + 1:03d}_{safe_name}.npy"
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, arr)
        saved_count += 1

        # Check for issues
        issues = []
        if np.any(np.isnan(arr)):
            issues.append("HAS NaN!")
        if np.any(np.isinf(arr)):
            issues.append("HAS Inf!")
        issue_str = f" *** {', '.join(issues)} ***" if issues else ""

        print(f"  [{i + 1:03d}] {name}: {list(arr.shape)} {arr.dtype}"
              f" range=[{arr.min():.4f}, {arr.max():.4f}]{issue_str}")

    print(f"\nSaved {saved_count} activation checkpoints + {len(feeds)} inputs")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # Verify determinism
    print("\n## Determinism Check")
    results2 = session.run(None, feeds)
    all_match = True
    for name, r1, r2 in zip(session_outputs, results, results2):
        if r1 is None or r2 is None:
            continue
        a1, a2 = np.array(r1), np.array(r2)
        if not np.array_equal(a1, a2):
            print(f"  WARNING: {name} is NOT deterministic (max diff: {np.max(np.abs(a1 - a2))})")
            all_match = False
    if all_match:
        print("  All activations are bitwise deterministic across runs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump ONNX model intermediate activations")
    parser.add_argument("model", help="Path to .onnx model file")
    parser.add_argument("--output-dir", "-o", default="models/activations",
                        help="Directory to save .npy files (default: models/activations)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for input generation (default: 42)")
    args = parser.parse_args()
    dump_activations(args.model, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
