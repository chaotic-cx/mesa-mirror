# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import logging
from typing import List, cast

from torch import fx, per_channel_symmetric, per_tensor_affine, qint8, qint32
import torch
from torchao.quantization.pt2e import (
    FixedQParamsObserver,
    HistogramObserver,
    PerChannelMinMaxObserver,
)

from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    OperatorConfig,
    Quantizer,
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
    annotate_input_qspec_map,
    annotate_output_qspec,
    get_module_name_filter,
)

from .dim_order_pass import merges_spatial_dims
from .exir_to_gallium import (
    FUSABLE_ACTIVATION_OPS_ATEN,
    activation_is_absorbable,
    is_2x_resize,
)

Q_ANNOTATION_KEY = "quantization_annotation"


def _shares_input_quantization(node: fx.Node) -> bool:
    output = node.meta.get("val")
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        return False
    target = node.target
    if target == torch.ops.aten.cat.default:
        return output.dim() in (3, 4)
    if target == torch.ops.aten.upsample_nearest2d.vec:
        return is_2x_resize(node)
    if target == torch.ops.aten.max_pool2d.default:
        return output.dim() == 4
    if target in (torch.ops.aten.view.default,
                  torch.ops.aten.flatten.using_ints):
        return merges_spatial_dims(node)
    if target == torch.ops.aten.permute.default:
        return output.dim() == 3
    return False


def _derive_conv_bias_qparams(obs_or_fqs):
    """Derive a convolution's bias quantisation from its input activation
    and weight observers: bias scale is input_scale * weight_scale[c] per
    output channel, zero point 0, as the accumulator it feeds expects."""
    act_obs, weight_obs = obs_or_fqs[0], obs_or_fqs[1]
    act_scale, _ = act_obs.calculate_qparams()
    weight_scale, _ = weight_obs.calculate_qparams()
    bias_scale = (act_scale.to(torch.float64) *
                  weight_scale.to(torch.float64)).flatten()
    return (bias_scale.to(torch.float32),
            torch.zeros_like(bias_scale, dtype=torch.int))

def is_annotated(node: fx.Node) -> bool:
    """Given a node return whether the node is annotated."""
    if Q_ANNOTATION_KEY not in node.meta:
        return False

    annotation = cast(QuantizationAnnotation, node.meta[Q_ANNOTATION_KEY])
    if not annotation._annotated:
        return False

    return True

def mark_node_as_annotated(node: fx.Node) -> None:
    """Marks node as annotated. If needed, an empty  QuantizationAnnotation is added
    to the quantization_annotation node meta entry.
    """
    if Q_ANNOTATION_KEY not in node.meta:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
    node.meta[Q_ANNOTATION_KEY]._annotated = True

class TorxQuantizer(Quantizer):

    def __init__(self):
        super().__init__()

        self.io_spec = QuantizationSpec(dtype=qint8,
                                        quant_min=-128,
                                        quant_max=127,
                                        observer_or_fake_quant_ctr=HistogramObserver,
                                        qscheme=per_tensor_affine)

        # Symmetric per-channel weights over [-127, 127]: excluding -128
        # keeps the weight zero point at 0, which the NPU requires and TFLite
        # does the same.  Per channel because a depthwise convolution's
        # channels span very different ranges.
        self.weight_spec = QuantizationSpec(
            dtype=qint8,
            quant_min=-127,
            quant_max=127,
            observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(
                eps=2 ** -16),
            qscheme=per_channel_symmetric,
            ch_axis=0)

        self.unit_interval_spec = QuantizationSpec(
            dtype=qint8,
            quant_min=-128,
            quant_max=127,
            observer_or_fake_quant_ctr=FixedQParamsObserver.with_args(
                scale=1.0 / 256.0, zero_point=-128),
            qscheme=per_tensor_affine)

        self.unquantized_module_filters = {}

    def set_module_unquantized(self, module_name: str) -> "TorxQuantizer":
        self.unquantized_module_filters[module_name] = (
            get_module_name_filter(module_name))
        return self

    def _is_unquantized(self, node: fx.Node) -> bool:
        return any(f(node) for f in self.unquantized_module_filters.values())

    def _annotate_io(
        self,
        model: fx.GraphModule,
    ):
        for node in model.graph.nodes:
            if is_annotated(node):
                continue
            if node.op == "placeholder" and len(node.users) > 0:
                annotate_output_qspec(
                    node,
                    self.io_spec,
                )
                mark_node_as_annotated(node)
            if node.op == "output":
                parent = node.all_input_nodes[0]
                if self._is_unquantized(parent):
                    continue
                annotate_input_qspec_map(
                    node, parent, self.io_spec
                )
                mark_node_as_annotated(node)

    def _annotate_with_optional_fused_act(self, node: fx.Node, input_qspec_map: dict) -> None:
        """Annotate node with input specs; if a single fusable activation follows,
        move the output Q/DQ annotation onto the activation node instead."""
        users = list(node.users.keys())
        act_node = None
        if (len(users) == 1 and users[0].target in FUSABLE_ACTIVATION_OPS_ATEN and
                not self._is_unquantized(users[0]) and
                activation_is_absorbable(
                    users[0], node.target == torch.ops.aten.conv2d.default)):
            act_node = users[0]

        if act_node is not None:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
            act_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                output_qspec=self.io_spec,
                _annotated=True,
            )
        else:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=self.io_spec,
                _annotated=True,
            )

    def _annotate_shared_io(self, node: fx.Node, inputs: list) -> None:
        shared = SharedQuantizationSpec((inputs[0], node))
        input_qspec_map = {inputs[0]: self.io_spec}
        for input_node in inputs[1:]:
            if input_node not in input_qspec_map:
                input_qspec_map[input_node] = shared
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=shared,
            _annotated=True,
        )

    def _sigmoid_output_spec(self, node: fx.Node) -> QuantizationSpec:
        """Return sigmoid's output spec, pinned to [0, 1] unless a consumer
        would make other tensors share that range."""
        if any(_shares_input_quantization(user) for user in node.users):
            return self.io_spec
        return self.unit_interval_spec

    def _annotate_operations(self, model: fx.GraphModule):
        for node in model.graph.nodes:
            if is_annotated(node) or self._is_unquantized(node):
                continue
            if node.op == "call_function":
                if node.target == torch.ops.aten.conv2d.default:
                    input_qspec_map = {}
                    input_act = node.args[0]
                    assert isinstance(input_act, fx.Node)
                    input_qspec_map[input_act] = self.io_spec

                    weight = node.args[1]
                    assert isinstance(weight, fx.Node)
                    input_qspec_map[weight] = self.weight_spec

                    if len(node.args) > 2:
                        bias = node.args[2]
                        if isinstance(bias, fx.Node):
                            input_qspec_map[bias] = DerivedQuantizationSpec(
                                derived_from=[(input_act, node),
                                              (weight, node)],
                                derive_qparams_fn=_derive_conv_bias_qparams,
                                dtype=qint32,
                                quant_min=-(2 ** 31) + 1,
                                quant_max=2 ** 31 - 1,
                                qscheme=per_channel_symmetric,
                                ch_axis=0)

                    # Fused conv+activation: output Q/DQ goes after the
                    # activation so the observer sees clamped values.
                    self._annotate_with_optional_fused_act(node, input_qspec_map)

                elif node.target in (torch.ops.aten.add.Tensor,
                                     torch.ops.aten.add_.Tensor):
                    input_qspec_map = {}
                    input_qspec_map[node.args[0]] = self.io_spec
                    input_qspec_map[node.args[1]] = self.io_spec

                    self._annotate_with_optional_fused_act(node, input_qspec_map)

                elif node.target == torch.ops.aten.cat.default:
                    if _shares_input_quantization(node):
                        self._annotate_shared_io(node, list(node.args[0]))

                elif node.target == torch.ops.aten.upsample_nearest2d.vec:
                    if _shares_input_quantization(node):
                        self._annotate_shared_io(node, [node.args[0]])

                elif node.target in (torch.ops.aten.sigmoid.default,
                                     torch.ops.aten.sigmoid_.default):
                    output = node.meta.get("val")
                    if isinstance(output, torch.Tensor) and output.dim() == 4:
                        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                            input_qspec_map={node.args[0]: self.io_spec},
                            output_qspec=self._sigmoid_output_spec(node),
                            _annotated=True,
                        )

                elif node.target == torch.ops.aten.max_pool2d.default:
                    if _shares_input_quantization(node):
                        self._annotate_shared_io(node, [node.args[0]])

                elif node.target in (torch.ops.aten.view.default,
                                     torch.ops.aten.flatten.using_ints,
                                     torch.ops.aten.permute.default):
                    if _shares_input_quantization(node):
                        self._annotate_shared_io(node, [node.args[0]])

    def annotate(self, model):
        for module_name, module_filter in (
                self.unquantized_module_filters.items()):
            if not any(module_filter(node) for node in model.graph.nodes):
                raise ValueError(
                    f"No node of the model belongs to module {module_name}")

        # Annotate operations first, then only add I/O quantisation if at
        # least one operation was annotated.  This avoids triggering the
        # expensive convert_pt2e() path for graphs that contain only
        # unsupported ops (e.g. standalone BatchNorm layers), saving ~5 s
        # per such test.
        self._annotate_operations(model)

        has_annotated_op = any(
            is_annotated(node)
            for node in model.graph.nodes
            if node.op == "call_function"
        )
        if has_annotated_op:
            self._annotate_io(model)

        return model

    def validate(self, model: fx.GraphModule) -> None:
        for node in model.graph.nodes:
            if Q_ANNOTATION_KEY not in node.meta:
                logging.debug(f"Node {node.name} is not annotated for quantization.")

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []
