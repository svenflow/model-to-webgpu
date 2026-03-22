#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tensorflow", "numpy"]
# ///
"""TFLite Model Inspector — Phase 1 starter template.

Parses a TFLite model and prints:
  - Operation inventory (grouped by type with counts)
  - Tensor catalogue (name, shape, dtype, quantization params, byte size)
  - Model inputs and outputs with shapes
  - Subgraph structure
  - Quantization details

Usage:
    ./inspect_tflite.py model.tflite
    ./inspect_tflite.py model.tflite --verbose

Note: If the model is inside a MediaPipe .task bundle, extract the .tflite
file from the zip first.
"""
import argparse
import zipfile
import sys
import os
import tempfile
from collections import Counter

import numpy as np
import tensorflow as tf


DTYPE_MAP = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
}


def extract_from_task_bundle(path: str) -> str | None:
    """Extract .tflite from a MediaPipe .task zip bundle."""
    if not zipfile.is_zipfile(path):
        return None
    with zipfile.ZipFile(path, "r") as zf:
        tflite_files = [n for n in zf.namelist() if n.endswith(".tflite")]
        if not tflite_files:
            return None
        print(f"Found .tflite inside .task bundle: {tflite_files}")
        tmp = tempfile.mkdtemp()
        extracted = zf.extract(tflite_files[0], tmp)
        return extracted


def inspect(model_path: str, verbose: bool = False) -> None:
    # Handle .task bundles
    actual_path = model_path
    if model_path.endswith(".task"):
        extracted = extract_from_task_bundle(model_path)
        if extracted:
            actual_path = extracted
            print(f"Extracted TFLite from .task bundle\n")
        else:
            print("WARNING: .task file is not a valid zip or contains no .tflite")
            return

    interpreter = tf.lite.Interpreter(model_path=actual_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()

    print(f"Model: {model_path}")
    print(f"Tensors: {len(tensor_details)}")
    print()

    # --- Inputs ---
    print("## Inputs")
    for inp in input_details:
        quant = inp.get("quantization_parameters", {})
        scales = quant.get("scales", np.array([]))
        zps = quant.get("zero_points", np.array([]))
        quant_str = ""
        if len(scales) > 0 and scales[0] != 0.0:
            quant_str = f" [quantized: scale={scales[0]:.6f}, zp={zps[0]}]"
        print(f"  [{inp['index']:3d}] {inp['name']}: {list(inp['shape'])} {inp['dtype'].__name__}{quant_str}")
    print()

    # --- Outputs ---
    print("## Outputs")
    for out in output_details:
        quant = out.get("quantization_parameters", {})
        scales = quant.get("scales", np.array([]))
        zps = quant.get("zero_points", np.array([]))
        quant_str = ""
        if len(scales) > 0 and scales[0] != 0.0:
            quant_str = f" [quantized: scale={scales[0]:.6f}, zp={zps[0]}]"
        print(f"  [{out['index']:3d}] {out['name']}: {list(out['shape'])} {out['dtype'].__name__}{quant_str}")
    print()

    # --- Operation inventory ---
    # TFLite doesn't expose ops directly through the Python API,
    # so we read the flatbuffer to get op codes
    with open(actual_path, "rb") as f:
        model_bytes = f.read()

    try:
        from tensorflow.lite.python import schema_py_generated as schema
        buf = schema.Model.GetRootAs(model_bytes, 0)
        subgraph_count = buf.SubgraphsLength()

        print(f"## Subgraphs: {subgraph_count}")
        for sg_idx in range(subgraph_count):
            sg = buf.Subgraphs(sg_idx)
            name = sg.Name().decode() if sg.Name() else f"subgraph_{sg_idx}"
            op_count = sg.OperatorsLength()
            print(f"  {name}: {op_count} operators")

            # Count op types
            op_codes = []
            for op_idx in range(op_count):
                op = sg.Operators(op_idx)
                opcode_idx = op.OpcodeIndex()
                builtin = buf.OperatorCodes(opcode_idx)
                # BuiltinCode gives the deprecated code, DeprecatedBuiltinCode gives the old one
                code = builtin.DeprecatedBuiltinCode()
                if code == 127:  # PLACEHOLDER_FOR_GREATER_OP_CODES
                    code = builtin.BuiltinCode()
                op_codes.append(code)

            op_counter = Counter(op_codes)
            # Map codes to names using TFLite builtins
            builtin_names = {v: k for k, v in vars(tf.lite.experimental).items() if isinstance(v, int)} if hasattr(tf.lite, "experimental") else {}
            for code, count in op_counter.most_common():
                name = builtin_names.get(code, f"OP_{code}")
                print(f"    {name}: {count}")
        print()
    except Exception as e:
        print(f"  (Could not parse flatbuffer ops: {e})")
        print()

    # --- Tensor catalogue ---
    input_indices = {d["index"] for d in input_details}
    output_indices = {d["index"] for d in output_details}
    weight_count = 0
    total_bytes = 0

    print("## Weight Tensors")
    for t in tensor_details:
        idx = t["index"]
        if idx in input_indices or idx in output_indices:
            continue

        shape = list(t["shape"])
        dtype = t["dtype"]
        quant = t.get("quantization_parameters", {})
        scales = quant.get("scales", np.array([]))
        zps = quant.get("zero_points", np.array([]))

        # Estimate byte size
        elem_count = int(np.prod(shape)) if len(shape) > 0 else 0
        bytes_per_elem = dtype.itemsize if hasattr(dtype, "itemsize") else np.dtype(dtype).itemsize
        byte_size = elem_count * bytes_per_elem
        total_bytes += byte_size

        quant_str = ""
        if len(scales) > 0 and scales[0] != 0.0:
            quant_str = f" [per-{'axis' if len(scales) > 1 else 'tensor'} quantized]"
            weight_count += 1
        elif len(shape) > 0 and elem_count > 0:
            weight_count += 1

        if verbose or (len(shape) > 0 and elem_count > 1):
            print(f"  [{idx:3d}] {t['name']}: {shape} {dtype.__name__} ({byte_size:,} bytes){quant_str}")

    print(f"\n  Weight tensors: {weight_count}")
    print(f"  Total weight size: {total_bytes / 1024 / 1024:.1f} MB")
    print()

    # --- Component detection ---
    prefixes: dict[str, int] = {}
    for t in tensor_details:
        name = t["name"]
        parts = name.split("/")
        if len(parts) >= 2:
            component = parts[0]
            prefixes[component] = prefixes.get(component, 0) + 1

    if prefixes:
        print("## Components (by tensor name prefix)")
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:20]:
            print(f"  {prefix}: {count} tensors")
        print()

    print("## Notes")
    print("  - Check for LITE vs FULL variant: architectures differ significantly")
    print("  - Check for fused BatchNorm (BN folded into conv bias)")
    print("  - NHWC layout: transpose weights to NCHW for WebGPU shaders")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TFLite model structure")
    parser.add_argument("model", help="Path to .tflite or .task model file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all tensors including scalars")
    args = parser.parse_args()
    inspect(args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
