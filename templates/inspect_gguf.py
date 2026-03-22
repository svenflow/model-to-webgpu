#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""GGUF Model Inspector — Phase 1 starter template.

Parses a GGUF file header and prints:
  - File metadata (version, tensor count, KV count)
  - All key-value metadata entries
  - Model config (hidden_size, num_layers, head_count, etc.)
  - Tokenizer info (vocab size, special tokens)
  - Tensor inventory (name, shape, quantization type, byte size)

Usage:
    ./inspect_gguf.py model.gguf
    ./inspect_gguf.py model.gguf --verbose
    ./inspect_gguf.py model.gguf --tensors-only
"""
import argparse
import struct
import sys
from collections import Counter

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_VERSION_3 = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML quantization types
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "IQ2_XXS",
    16: "IQ2_XS",
    17: "IQ3_XXS",
    18: "IQ1_S",
    19: "IQ4_NL",
    20: "IQ3_S",
    21: "IQ2_S",
    22: "IQ4_XS",
    28: "I8",
    29: "I16",
    30: "I32",
    31: "I64",
    32: "F64",
    33: "IQ1_M",
}

# Block sizes for quantized types (bytes per block of 32 elements)
GGML_BLOCK_SIZE = {
    "F32": (1, 4),      # 1 element, 4 bytes
    "F16": (1, 2),      # 1 element, 2 bytes
    "Q4_0": (32, 18),   # 32 elements, 18 bytes (2B scale + 16B quants)
    "Q4_1": (32, 20),   # 32 elements, 20 bytes
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),   # 32 elements, 34 bytes (2B scale + 32B quants)
    "Q8_1": (32, 36),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "I8": (1, 1),
    "I16": (1, 2),
    "I32": (1, 4),
}


class GGUFReader:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        self.metadata: dict[str, object] = {}
        self.tensors: list[dict] = []
        self._parse_header()

    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) < size:
            raise EOFError(f"Expected {size} bytes, got {len(data)}")
        return struct.unpack(fmt, data)

    def _read_string(self) -> str:
        (length,) = self._read("<Q")
        data = self.f.read(length)
        return data.decode("utf-8", errors="replace")

    def _read_value(self, vtype: int):
        if vtype == GGUF_TYPE_UINT8:
            return self._read("<B")[0]
        elif vtype == GGUF_TYPE_INT8:
            return self._read("<b")[0]
        elif vtype == GGUF_TYPE_UINT16:
            return self._read("<H")[0]
        elif vtype == GGUF_TYPE_INT16:
            return self._read("<h")[0]
        elif vtype == GGUF_TYPE_UINT32:
            return self._read("<I")[0]
        elif vtype == GGUF_TYPE_INT32:
            return self._read("<i")[0]
        elif vtype == GGUF_TYPE_FLOAT32:
            return self._read("<f")[0]
        elif vtype == GGUF_TYPE_BOOL:
            return bool(self._read("<B")[0])
        elif vtype == GGUF_TYPE_STRING:
            return self._read_string()
        elif vtype == GGUF_TYPE_ARRAY:
            (elem_type,) = self._read("<I")
            (count,) = self._read("<Q")
            return [self._read_value(elem_type) for _ in range(count)]
        elif vtype == GGUF_TYPE_UINT64:
            return self._read("<Q")[0]
        elif vtype == GGUF_TYPE_INT64:
            return self._read("<q")[0]
        elif vtype == GGUF_TYPE_FLOAT64:
            return self._read("<d")[0]
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _parse_header(self):
        magic, version, tensor_count, kv_count = self._read("<IIQQ")

        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic: 0x{magic:08X})")

        self.version = version
        self.tensor_count = tensor_count
        self.kv_count = kv_count

        # Read KV metadata
        for _ in range(kv_count):
            key = self._read_string()
            (vtype,) = self._read("<I")
            value = self._read_value(vtype)
            self.metadata[key] = value

        # Read tensor info
        for _ in range(tensor_count):
            name = self._read_string()
            (n_dims,) = self._read("<I")
            dims = [self._read("<Q")[0] for _ in range(n_dims)]
            (qtype,) = self._read("<I")
            (offset,) = self._read("<Q")

            type_name = GGML_TYPE_NAMES.get(qtype, f"UNKNOWN_{qtype}")
            elem_count = 1
            for d in dims:
                elem_count *= d

            # Calculate byte size
            if type_name in GGML_BLOCK_SIZE:
                block_elems, block_bytes = GGML_BLOCK_SIZE[type_name]
                num_blocks = (elem_count + block_elems - 1) // block_elems
                byte_size = num_blocks * block_bytes
            else:
                byte_size = 0

            self.tensors.append({
                "name": name,
                "shape": dims,
                "type": type_name,
                "type_id": qtype,
                "offset": offset,
                "elements": elem_count,
                "bytes": byte_size,
            })

    def close(self):
        self.f.close()


def format_value(value) -> str:
    if isinstance(value, list):
        if len(value) > 8:
            preview = ", ".join(str(v) for v in value[:4])
            return f"[{preview}, ... ({len(value)} items)]"
        return str(value)
    return str(value)


def inspect(model_path: str, verbose: bool = False, tensors_only: bool = False) -> None:
    reader = GGUFReader(model_path)

    print(f"Model: {model_path}")
    print(f"GGUF version: {reader.version}")
    print(f"Tensor count: {reader.tensor_count}")
    print(f"KV metadata entries: {reader.kv_count}")
    print()

    if not tensors_only:
        # --- Model config ---
        config_keys = [k for k in reader.metadata if not k.startswith("tokenizer.")]
        print("## Model Config")
        for key in sorted(config_keys):
            value = reader.metadata[key]
            print(f"  {key}: {format_value(value)}")
        print()

        # --- Tokenizer info ---
        tokenizer_keys = [k for k in reader.metadata if k.startswith("tokenizer.")]
        if tokenizer_keys:
            print("## Tokenizer")
            for key in sorted(tokenizer_keys):
                value = reader.metadata[key]
                if key == "tokenizer.ggml.tokens":
                    print(f"  {key}: [{len(value)} tokens]")
                    if verbose:
                        for i, tok in enumerate(value[:20]):
                            print(f"    [{i}] {repr(tok)}")
                        if len(value) > 20:
                            print(f"    ... ({len(value) - 20} more)")
                elif key == "tokenizer.ggml.scores":
                    print(f"  {key}: [{len(value)} scores]")
                elif key == "tokenizer.ggml.token_type":
                    print(f"  {key}: [{len(value)} types]")
                elif key == "tokenizer.ggml.merges":
                    print(f"  {key}: [{len(value)} merges]")
                else:
                    print(f"  {key}: {format_value(value)}")
            print()

    # --- Tensor inventory ---
    total_bytes = sum(t["bytes"] for t in reader.tensors)
    quant_types = Counter(t["type"] for t in reader.tensors)

    print("## Quantization Summary")
    for qtype, count in quant_types.most_common():
        print(f"  {qtype}: {count} tensors")
    print()

    print("## Tensors")
    for t in reader.tensors:
        print(f"  {t['name']}: {t['shape']} {t['type']} ({t['bytes']:,} bytes)")
    print(f"\n  Total tensor data: {total_bytes / 1024 / 1024:.1f} MB")
    print()

    # --- Component detection ---
    prefixes: dict[str, int] = {}
    for t in reader.tensors:
        parts = t["name"].split(".")
        if len(parts) >= 2:
            component = ".".join(parts[:2])
            prefixes[component] = prefixes.get(component, 0) + 1

    if prefixes:
        print("## Components (by tensor name prefix)")
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
            print(f"  {prefix}: {count} tensors")
        print()

    reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GGUF model structure")
    parser.add_argument("model", help="Path to .gguf model file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print tokenizer tokens and extra detail")
    parser.add_argument("--tensors-only", "-t", action="store_true", help="Only print tensor info")
    args = parser.parse_args()
    inspect(args.model, verbose=args.verbose, tensors_only=args.tensors_only)


if __name__ == "__main__":
    main()
