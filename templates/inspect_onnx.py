#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["onnx", "onnxruntime", "numpy"]
# ///
"""ONNX Model Inspector — Phase 1 starter template.

Parses an ONNX model and prints:
  - Operation inventory (grouped by type with counts)
  - Weight tensor catalogue (name, shape, dtype, quantization, byte size)
  - Graph inputs and outputs with shapes
  - Named components / blocks
  - Forward-pass data flow summary

Usage:
    ./inspect_onnx.py model.onnx
    ./inspect_onnx.py model.onnx --verbose
"""
import argparse
import sys
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto


def dtype_name(dt: int) -> str:
    return TensorProto.DataType.Name(dt)


def shape_from_type(type_proto) -> list:
    if not type_proto.HasField("tensor_type"):
        return []
    return [
        d.dim_value if d.dim_value else d.dim_param
        for d in type_proto.tensor_type.shape.dim
    ]


def inspect(model_path: str, verbose: bool = False) -> None:
    model = onnx.load(model_path)
    graph = model.graph

    print(f"Model: {model_path}")
    print(f"IR version: {model.ir_version}")
    print(f"Opset: {', '.join(f'{o.domain or \"ai.onnx\"}:{o.version}' for o in model.opset_import)}")
    print(f"Nodes: {len(graph.node)}")
    print(f"Initializers: {len(graph.initializer)}")
    print()

    # --- Operation inventory ---
    ops = Counter(n.op_type for n in graph.node)
    print("## Operations")
    for op, count in ops.most_common():
        print(f"  {op}: {count}")
    print()

    # --- Weight inventory ---
    total_bytes = 0
    print("## Weights")
    for init in graph.initializer:
        shape = list(init.dims)
        dt = dtype_name(init.data_type)
        raw_size = len(init.raw_data) if init.raw_data else 0
        total_bytes += raw_size

        # Detect quantization hints
        quant_info = ""
        if init.data_type in (TensorProto.INT8, TensorProto.UINT8):
            quant_info = " [quantized]"
        elif init.data_type == TensorProto.FLOAT16:
            quant_info = " [f16]"

        print(f"  {init.name}: {shape} {dt} ({raw_size:,} bytes){quant_info}")

    print(f"\n  Total weight size: {total_bytes / 1024 / 1024:.1f} MB")
    print()

    # --- Graph inputs ---
    init_names = {init.name for init in graph.initializer}
    print("## Inputs (non-weight)")
    for inp in graph.input:
        if inp.name not in init_names:
            shape = shape_from_type(inp.type)
            print(f"  {inp.name}: {shape}")
    print()

    print("## Outputs")
    for out in graph.output:
        shape = shape_from_type(out.type)
        print(f"  {out.name}: {shape}")
    print()

    # --- Component detection ---
    prefixes: dict[str, int] = {}
    for node in graph.node:
        parts = node.name.split("/")
        if len(parts) >= 2:
            component = "/".join(parts[:2])
            prefixes[component] = prefixes.get(component, 0) + 1

    if prefixes:
        print("## Components (by node name prefix)")
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
            print(f"  {prefix}: {count} nodes")
        print()

    # --- Quantization detection ---
    quant_ops = [n for n in graph.node if "Quantize" in n.op_type or "Integer" in n.op_type]
    if quant_ops:
        print("## Quantized Operations")
        quant_types = Counter(n.op_type for n in quant_ops)
        for op, count in quant_types.most_common():
            print(f"  {op}: {count}")
        print()

    # --- Verbose: full node list ---
    if verbose:
        print("## All Nodes (in execution order)")
        for i, node in enumerate(graph.node):
            inputs = ", ".join(node.input[:3])
            if len(node.input) > 3:
                inputs += f", ... (+{len(node.input) - 3})"
            outputs = ", ".join(node.output[:2])
            print(f"  [{i:4d}] {node.op_type:30s} {node.name}")
            print(f"         in:  {inputs}")
            print(f"         out: {outputs}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ONNX model structure")
    parser.add_argument("model", help="Path to .onnx model file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full node list")
    args = parser.parse_args()
    inspect(args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
