#!/usr/bin/env python3
#
# Copyright 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
#
# SPDX-License-Identifier: MIT

"""Usage:
  split_tflite_tests.py split <model.tflite> <output_dir>
  split_tflite_tests.py cut <model.tflite> <output.tflite> <ranges>...
"""

import argparse
import struct
import sys
from pathlib import Path

import flatbuffers
from flatbuffers import number_types as N


# Just to not depend on the python bindings for TFLite.
# To be removed once it's generally packaged in distros.
LAYOUTS = {
    "Model": {
        "version": (0, "uint32"),
        "operator_codes": (1, "[OperatorCode]"),
        "subgraphs": (2, "[SubGraph]"),
        "description": (3, "string"),
        "buffers": (4, "[Buffer]"),
        "metadata_buffer": (5, None),
        "metadata": (6, None),
        "signature_defs": (7, "[SignatureDef]"),
    },
    "OperatorCode": {
        "deprecated_builtin_code": (0, "int8"),
        "custom_code": (1, "string"),
        "version": (2, "int32"),
        "builtin_code": (3, "int32"),
    },
    "SubGraph": {
        "tensors": (0, "[Tensor]"),
        "inputs": (1, "[int32]"),
        "outputs": (2, "[int32]"),
        "operators": (3, "[Operator]"),
        "name": (4, "string"),
    },
    "Tensor": {
        "shape": (0, "[int32]"),
        "type": (1, "int8"),
        "buffer": (2, "uint32"),
        "name": (3, "string"),
        "quantization": (4, "QuantizationParameters"),
        "is_variable": (5, "bool"),
        "shape_signature": (7, "[int32]"),
        "has_rank": (8, "bool"),
    },
    "QuantizationParameters": {
        "min": (0, "[float32]"),
        "max": (1, "[float32]"),
        "scale": (2, "[float32]"),
        "zero_point": (3, "[int64]"),
        "quantized_dimension": (6, "int32"),
    },
    "Buffer": {
        "data": (0, "[uint8]"),
        "offset": (1, "uint64"),
        "size": (2, "uint64"),
    },
    "Operator": {
        "opcode_index": (0, "uint32"),
        "inputs": (1, "[int32]"),
        "outputs": (2, "[int32]"),
        "builtin_options_type": (3, "uint8"),
        "builtin_options": (4, "union BuiltinOptions"),
        "custom_options": (5, "[uint8]"),
        "custom_options_format": (6, "int8"),
        "mutating_variable_inputs": (7, "[bool]"),
    },
    "SignatureDef": {
        "inputs": (0, "[TensorMap]"),
        "outputs": (1, "[TensorMap]"),
        "signature_key": (2, "string"),
        "deprecated_tag": (3, "string"),
        "subgraph_index": (4, "uint32"),
    },
    "TensorMap": {
        "name": (0, "string"),
        "tensor_index": (1, "uint32"),
    },
    "Conv2DOptions": {
        "padding": (0, "int8"),
        "stride_w": (1, "int32"),
        "stride_h": (2, "int32"),
        "fused_activation_function": (3, "int8"),
        "dilation_w_factor": (4, "int32"),
        "dilation_h_factor": (5, "int32"),
        "quantized_bias_type": (6, "int8"),
    },
    "DepthwiseConv2DOptions": {
        "padding": (0, "int8"),
        "stride_w": (1, "int32"),
        "stride_h": (2, "int32"),
        "depth_multiplier": (3, "int32"),
        "fused_activation_function": (4, "int8"),
        "dilation_w_factor": (5, "int32"),
        "dilation_h_factor": (6, "int32"),
    },
    "Pool2DOptions": {
        "padding": (0, "int8"),
        "stride_w": (1, "int32"),
        "stride_h": (2, "int32"),
        "filter_width": (3, "int32"),
        "filter_height": (4, "int32"),
        "fused_activation_function": (5, "int8"),
    },
    "FullyConnectedOptions": {
        "fused_activation_function": (0, "int8"),
        "weights_format": (1, "int8"),
        "keep_num_dims": (2, "bool"),
        "asymmetric_quantize_inputs": (3, "bool"),
        "quantized_bias_type": (4, "int8"),
    },
    "SoftmaxOptions": {
        "beta": (0, "float32"),
    },
    "ConcatenationOptions": {
        "axis": (0, "int32"),
        "fused_activation_function": (1, "int8"),
    },
    "AddOptions": {
        "fused_activation_function": (0, "int8"),
        "pot_scale_int16": (1, "bool"),
    },
    "ResizeBilinearOptions": {
        "new_height": (0, "int32"),
        "new_width": (1, "int32"),
        "align_corners": (2, "bool"),
        "half_pixel_centers": (3, "bool"),
    },
    "ReshapeOptions": {
        "new_shape": (0, "[int32]"),
    },
    "MulOptions": {
        "fused_activation_function": (0, "int8"),
    },
    "PadOptions": {},
    "SubOptions": {
        "fused_activation_function": (0, "int8"),
        "pot_scale_int16": (1, "bool"),
    },
    "DivOptions": {
        "fused_activation_function": (0, "int8"),
    },
    "StridedSliceOptions": {
        "begin_mask": (0, "int32"),
        "end_mask": (1, "int32"),
        "ellipsis_mask": (2, "int32"),
        "new_axis_mask": (3, "int32"),
        "shrink_axis_mask": (4, "int32"),
        "offset": (5, "bool"),
    },
    "ArgMaxOptions": {
        "output_type": (0, "int8"),
    },
    "PackOptions": {
        "values_count": (0, "int32"),
        "axis": (1, "int32"),
    },
    "UnpackOptions": {
        "num": (0, "int32"),
        "axis": (1, "int32"),
    },
    "ResizeNearestNeighborOptions": {
        "align_corners": (0, "bool"),
        "half_pixel_centers": (1, "bool"),
    },
    "LeakyReluOptions": {
        "alpha": (0, "float32"),
    },
    "ScatterNdOptions": {},
}

UNIONS = {
    "BuiltinOptions": {
        1: "Conv2DOptions",
        2: "DepthwiseConv2DOptions",
        5: "Pool2DOptions",
        8: "FullyConnectedOptions",
        9: "SoftmaxOptions",
        10: "ConcatenationOptions",
        11: "AddOptions",
        15: "ResizeBilinearOptions",
        17: "ReshapeOptions",
        21: "MulOptions",
        22: "PadOptions",
        28: "SubOptions",
        29: "DivOptions",
        32: "StridedSliceOptions",
        40: "ArgMaxOptions",
        59: "PackOptions",
        64: "UnpackOptions",
        74: "ResizeNearestNeighborOptions",
        75: "LeakyReluOptions",
        97: "ScatterNdOptions",
    },
}

SCALARS = {
    "bool": ("<?", N.BoolFlags),
    "int8": ("<b", N.Int8Flags),
    "uint8": ("<B", N.Uint8Flags),
    "int32": ("<i", N.Int32Flags),
    "uint32": ("<I", N.Uint32Flags),
    "int64": ("<q", N.Int64Flags),
    "uint64": ("<Q", N.Uint64Flags),
    "float32": ("<f", N.Float32Flags),
}

BUILTIN_CUSTOM = 32
OPTIONAL_TENSOR = -1


class TFLiteError(Exception):
    pass


def _u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def _read_vector(data, field, kind):
    vector = field + _u32(data, field)
    count = _u32(data, vector)
    start = vector + 4
    element = kind[1:-1]

    if element in ("uint8", "float32"):
        size = struct.calcsize(SCALARS[element][0])
        return bytes(data[start:start + count * size])

    if element in SCALARS:
        fmt = SCALARS[element][0]
        return list(struct.unpack_from(f"<{count}{fmt[1:]}", data, start))

    tables = []
    for i in range(count):
        offset = start + i * 4
        try:
            tables.append(_read_table(data, offset + _u32(data, offset),
                                      element))
        except TFLiteError as e:
            raise TFLiteError(f"{element} {i}: {e}") from None
    return tables


def _read_table(data, table, name):
    layout = LAYOUTS[name]
    slots = {slot: (field, kind) for field, (slot, kind) in layout.items()}
    vtable = table - struct.unpack_from("<i", data, table)[0]
    vtable_size = struct.unpack_from("<H", data, vtable)[0]
    obj = {}

    for slot in range((vtable_size - 4) // 2):
        offset = struct.unpack_from("<H", data, vtable + 4 + slot * 2)[0]
        if offset == 0:
            continue
        if slot not in slots:
            raise TFLiteError(f"{name} field in slot {slot} is not known")

        field, kind = slots[slot]
        position = table + offset
        if kind is None:
            continue
        elif kind in SCALARS:
            obj[field] = struct.unpack_from(SCALARS[kind][0], data,
                                            position)[0]
        elif kind == "string":
            string = position + _u32(data, position)
            length = _u32(data, string)
            obj[field] = bytes(data[string + 4:string + 4 + length])
        elif kind.startswith("["):
            obj[field] = _read_vector(data, position, kind)
        elif kind.startswith("union "):
            union = kind.split()[1]
            union_type = obj.get(field + "_type", 0)
            if union_type not in UNIONS[union]:
                raise TFLiteError(f"{union} type {union_type} is not known")
            obj[field] = _read_table(data, position + _u32(data, position),
                                     UNIONS[union][union_type])
        else:
            obj[field] = _read_table(data, position + _u32(data, position),
                                     kind)

    return obj


def read_model(data):
    return _read_table(data, _u32(data, 0), "Model")


def _write_vector(builder, kind, value):
    element = kind[1:-1]

    if element in SCALARS:
        fmt = SCALARS[element][0]
        size = struct.calcsize(fmt)
        if isinstance(value, bytes):
            raw = value
        else:
            raw = struct.pack(f"<{len(value)}{fmt[1:]}", *value)
        alignment = 16 if element == "uint8" else size
        builder.StartVector(size, len(raw) // size, alignment)
        builder.head = builder.head - len(raw)
        builder.Bytes[builder.head:builder.head + len(raw)] = raw
        return builder.EndVector()

    offsets = [_write_table(builder, element, obj) for obj in value]
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _write_table(builder, name, obj):
    layout = LAYOUTS[name]
    offsets = {}

    for field, value in obj.items():
        kind = layout[field][1]
        if kind in SCALARS:
            continue
        elif kind == "string":
            offsets[field] = builder.CreateString(value)
        elif kind.startswith("["):
            offsets[field] = _write_vector(builder, kind, value)
        elif kind.startswith("union "):
            union = kind.split()[1]
            offsets[field] = _write_table(
                builder, UNIONS[union][obj[field + "_type"]], value)
        else:
            offsets[field] = _write_table(builder, kind, value)

    builder.StartObject(max((slot for slot, _ in layout.values()),
                            default=-1) + 1)
    for field, value in obj.items():
        slot, kind = layout[field]
        if field in offsets:
            builder.PrependUOffsetTRelativeSlot(slot, offsets[field], 0)
        else:
            builder.PrependSlot(SCALARS[kind][1], slot, value, None)
    return builder.EndObject()


def write_model(model):
    builder = flatbuffers.Builder(1024)
    builder.Finish(_write_table(builder, "Model", model),
                   file_identifier=b"TFL3")
    return builder.Output()


def _builtin_code(model, op):
    opcode = model["operator_codes"][op.get("opcode_index", 0)]
    return max(opcode.get("deprecated_builtin_code", 0),
               opcode.get("builtin_code", 0))


def _cut(model, indices):
    subgraph = model["subgraphs"][0]
    count = len(subgraph["operators"])
    if not indices:
        raise TFLiteError("no operators to cut")
    for i in indices:
        if not 0 <= i < count:
            raise TFLiteError(f"operator {i} is outside the model's "
                              f"{count} operators")
    ops = [subgraph["operators"][i] for i in indices]

    tensors = {}
    buffers = {0: 0}
    for op in ops:
        for t in op.get("inputs", []) + op.get("outputs", []):
            if t == OPTIONAL_TENSOR:
                continue
            tensors.setdefault(t, len(tensors))
            buffers.setdefault(subgraph["tensors"][t].get("buffer", 0),
                               len(buffers))

    new_tensors = []
    for t in tensors:
        tensor = dict(subgraph["tensors"][t])
        tensor["buffer"] = buffers[tensor.get("buffer", 0)]
        new_tensors.append(tensor)
    new_buffers = [model["buffers"][b] for b in buffers]

    new_ops = []
    for op in ops:
        op = dict(op)
        for io in ("inputs", "outputs"):
            if io in op:
                op[io] = [t if t == OPTIONAL_TENSOR else tensors[t]
                          for t in op[io]]
        new_ops.append(op)

    def is_empty(buffer):
        return ("data" not in buffer and not buffer.get("offset", 0) and
                not buffer.get("size", 0))

    inputs = [t for t in new_ops[0].get("inputs", [])
              if t != OPTIONAL_TENSOR and
              is_empty(new_buffers[new_tensors[t]["buffer"]])]

    outputs = [t for op in new_ops for t in op.get("outputs", [])
               if t != OPTIONAL_TENSOR]
    for op in new_ops:
        for t in op.get("inputs", []):
            if t in outputs:
                outputs.remove(t)

    new_subgraph = dict(subgraph)
    new_subgraph.update(tensors=new_tensors, operators=new_ops,
                        inputs=inputs, outputs=outputs)

    cut = dict(model)
    cut["subgraphs"] = [new_subgraph] + model["subgraphs"][1:]
    cut["buffers"] = new_buffers

    if model.get("signature_defs"):
        def tensor_map(t):
            entry = {"tensor_index": t}
            if "name" in new_tensors[t]:
                entry["name"] = new_tensors[t]["name"]
            return entry

        signature = dict(model["signature_defs"][0])
        signature["inputs"] = [tensor_map(t) for t in inputs]
        signature["outputs"] = [tensor_map(t) for t in outputs]
        cut["signature_defs"] = [signature] + model["signature_defs"][1:]

    return write_model(cut)


def cut_model(data, indices):
    return _cut(read_model(data), indices)


def _remove_stale_cases(output_dir, case_names):
    for entry in output_dir.iterdir():
        if (entry.suffix == ".tflite" and entry.stem.isdigit()
                and entry.name not in case_names):
            entry.unlink()


def split_model(model_path, output_dir):
    data = model_path.read_bytes()
    model = read_model(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_names = set()
    for i, op in enumerate(model["subgraphs"][0]["operators"]):
        if _builtin_code(model, op) == BUILTIN_CUSTOM:
            continue
        case_name = f"{i:04d}.tflite"
        (output_dir / case_name).write_bytes(_cut(model, [i]))
        case_names.add(case_name)

    _remove_stale_cases(output_dir, case_names)

    (output_dir / "whole.tflite").write_bytes(data)


def parse_ranges(ranges):
    indices = []
    for r in ranges:
        if "-" in r:
            start, end = r.split("-")
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(r))
    return indices


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("model", type=Path)
    split.add_argument("output_dir", type=Path)

    cut = subparsers.add_parser("cut")
    cut.add_argument("model", type=Path)
    cut.add_argument("output", type=Path)
    cut.add_argument("ranges", nargs="+")

    args = parser.parse_args()

    try:
        if args.mode == "split":
            split_model(args.model, args.output_dir)
        else:
            args.output.write_bytes(
                cut_model(args.model.read_bytes(), parse_ranges(args.ranges)))
    except TFLiteError as e:
        sys.exit(f"{args.model}: {e}")


if __name__ == "__main__":
    main()
