# Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
#
# SPDX-License-Identifier: MIT

"""Create one test case per operation in model.

Produces per test case:

  <output_dir>/<model>/<op>/op.pt2                  -- per-op ExportedProgram
  <output_dir>/<model>/<op>/example_input.pt        -- float input tensor(s)
  <output_dir>/<model>/<op>/float-output-{i}.data   -- float32 CPU reference(s)

Usage:
  python split_torx_tests.py <models_dir> <output_dir> <stamp_file>
"""

import shutil
import sys
import warnings
from pathlib import Path

import torch
from torch.export import export

# python3 -P leaves the script's directory off sys.path.
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from extract_subgraph import build_subgraph, is_dequantize  # noqa: E402

_EXCLUDED_OPS = {
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.new_zeros.default",
    "aten.full.default",
    "aten.arange.default",
    "aten.dropout.default",
    "aten.slice.Tensor",
    "aten.view_as.default",
    "aten.view.default",
    "aten.view_copy.default",
    "aten.unsqueeze.default",
    "aten.chunk.default",
    "aten.detach.default",
    "aten.to.device",
    "aten._assert_tensor_metadata.default",
    "aten.batch_norm.default",
    "aten.hardtanh_.default",
    "aten.adaptive_avg_pool2d.default",
    "aten.flatten.using_ints",
}

_CONVOLUTION_OPS = {
    "aten.conv1d.default",
    "aten.conv2d.default",
}


def _is_interesting_op(node):
    target = str(node.target)
    val = node.meta.get("val")
    return (node.op == "call_function" and target.startswith("aten.")
            and target not in _EXCLUDED_OPS
            and hasattr(val, "dtype") and val.dtype.is_floating_point)


def _configure_known_warning_filters() -> None:
    """Suppress known third-party warning noise with narrow filters only."""
    warnings.filterwarnings(
        "ignore",
        message=(
            r"The given buffer is not writable, and PyTorch does not support "
            r"non-writable tensors\..*"
        ),
        category=UserWarning,
        module=r"torch\.export\.pt2_archive\._package",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=FutureWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message=r"guard_size_oblivious will be removed\..*",
        category=FutureWarning,
        module=r"executorch\.exir\.tensor",
    )

_ACTIVATION_OPS = {
    "aten.relu.default",
    "aten.relu_.default",
    "aten.hardtanh.default",
    "aten.hardtanh_.default",
    "aten.clamp.default",
    "aten.clamp_.default",
    "aten.silu.default",
    "aten.silu_.default",
    "aten.sigmoid.default",
    "aten.tanh.default",
}


_PASSTHROUGH_OPS = {
    "aten.detach.default",
    "aten.clone.default",
}


def _activation_user(node):
    if len(node.users) != 1:
        return []
    u = next(iter(node.users))
    if str(u.target) in _ACTIVATION_OPS:
        return [u]
    if str(u.target) in _PASSTHROUGH_OPS:
        act = _activation_user(u)
        return [u] + act if act else []
    return []


def _convolution_chain(node):
    chain = [node]
    if len(node.users) == 1:
        u = next(iter(node.users))
        if str(u.target) == "aten.batch_norm.default":
            chain.append(u)
    chain.extend(_activation_user(chain[-1]))
    return chain


def _dequantize_args(nodes):
    dqs = []

    def collect(arg):
        if is_dequantize(arg):
            dqs.append(arg)

    for n in nodes:
        torch.fx.node.map_arg(n.args, collect)
        torch.fx.node.map_arg(n.kwargs, collect)
    return dqs


def split_model(pt2_path):
    """Load a whole-model .pt2 and split into per-op submodules.

    Returns a list of ``(index_name, module, example_inputs)`` tuples.
    """
    ep = torch.export.load(str(pt2_path))
    gm = ep.module()

    ops = []
    in_chain = set()
    idx = 0
    for node in gm.graph.nodes:
        if node in in_chain or not _is_interesting_op(node):
            continue

        selected = [node]
        if str(node.target) in _CONVOLUTION_OPS:
            selected = _convolution_chain(node)
        in_chain.update(selected)
        selected += _dequantize_args(selected)

        torch.manual_seed(42 + idx)
        try:
            mod, example_inputs, _ = build_subgraph(gm, set(selected))
        except Exception as e:
            print(f"  SKIP op {idx:03d} ({node.target}): {e}", file=sys.stderr)
        else:
            if example_inputs:
                ops.append((f"{idx:03d}", mod, tuple(example_inputs)))
        idx += 1

    return ops


def _remove_stale_cases(model_out_dir, op_names):
    if not model_out_dir.exists():
        return
    for entry in model_out_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit() and entry.name not in op_names:
            shutil.rmtree(entry)


def _flatten_outputs(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        result = []
        for item in x:
            result.extend(_flatten_outputs(item))
        return result
    return [torch.tensor(x)]


def main():
    _configure_known_warning_filters()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("stamp_file", type=Path)
    cli = parser.parse_args()
    models_dir = cli.models_dir
    output_root = cli.output_dir
    stamp_file = cli.stamp_file

    pt2_files = sorted(models_dir.glob("*.pt2"))
    if not pt2_files:
        print(f"No .pt2 files found in {models_dir}", file=sys.stderr)
        sys.exit(1)

    for pt2_file in pt2_files:
        model_name = pt2_file.stem

        ops = split_model(pt2_file)
        model_out_dir = output_root / model_name

        _remove_stale_cases(model_out_dir, {op_name for op_name, _, _ in ops})

        for op_name, mod, example_inputs in ops:
            output_dir = model_out_dir / op_name
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)

            try:
                ep = export(mod, example_inputs, strict=False)
            except Exception as e:
                print(f"    SKIP {model_name}/{op_name}: export failed: {e}",
                      file=sys.stderr)
                shutil.rmtree(output_dir)
                continue

            # Save per-op .pt2 and example input(s) (consumed by compile step).
            torch.export.save(ep, str(output_dir / "op.pt2"))
            torch.save(example_inputs, str(output_dir / "example_input.pt"))

            # Compute float CPU reference output(s).
            module = ep.module()
            with torch.no_grad():
                float_output = module(*example_inputs)
            flat_outs = _flatten_outputs(float_output)
            for i, out in enumerate(flat_outs):
                out.numpy().tofile(str(output_dir / f"float-output-{i}.data"))

    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text("done\n")


if __name__ == "__main__":
    main()
