# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm_gaudi.extension.ops as hpu_ops
import vllm_gaudi.ops.modelopt_csf as modelopt_csf
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

try:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
except ImportError:
    try:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
    except ImportError:

        class RoutedExperts:  # type: ignore[no-redef]
            """Sentinel for vLLM revisions without RoutedExperts."""


logger = init_logger(__name__)

_SUPPORTED_FP8_ALGOS = ("FP8", "FP8_PER_CHANNEL_PER_TOKEN")


def _checkpoint_weight_loader(layer: torch.nn.Module, extra_weight_attrs: dict):
    """Build a synchronized loader with Gaudi2 E4M3 range conversion."""

    weight_loader = extra_weight_attrs.get("weight_loader")
    if hasattr(layer, "weight_loader_v2"):
        weight_loader = layer.weight_loader_v2
    if weight_loader is None:
        raise ValueError("ModelOpt HPU linear requires a checkpoint weight loader")
    if hpu_ops.is_hpu_gaudi2:
        # ModelOpt stores E4M3FN values. Gaudi2 uses the OCP E4M3 range, so
        # halve FP8 values and double inverse scales while preserving W exactly.
        weight_loader = hpu_ops.gaudi_weight_wrapper(weight_loader)
    return hpu_ops.synced_weight_loader(weight_loader)


class HPUModelOptFp8Config(ModelOptFp8Config):
    """ModelOpt FP8 configuration backed by Gaudi FP8 MME kernels."""

    def __init__(
        self,
        quant_method: str,
        is_checkpoint_fp8_serialized: bool,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
    ) -> None:
        super().__init__(quant_method, is_checkpoint_fp8_serialized, kv_cache_quant_method, exclude_modules)
        if self.quant_method == "FP8":
            self.LinearMethodCls = HPUModelOptFp8LinearMethod
        elif self.quant_method == "FP8_PER_CHANNEL_PER_TOKEN":
            self.LinearMethodCls = HPUModelOptFp8PcPtLinearMethod
        else:
            raise ValueError("Unsupported ModelOpt FP8 quant_algo on Gaudi: "
                             f"{self.quant_method}. Supported: {', '.join(_SUPPORTED_FP8_ALGOS)}.")

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, (Attention, MLAAttention)):
            return self.KVCacheMethodCls(self)

        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        if "vision_tower" in prefix or "vision_model" in prefix or "vit_large_projector" in prefix:
            return UnquantizedLinearMethod()

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return self.LinearMethodCls(self)
        if isinstance(layer, RoutedExperts):
            raise ValueError("FP8 ModelOpt fused MoE quantization is not supported on Gaudi")
        return None


class HPUModelOptFp8LinearMethod(LinearMethodBase):
    """Static per-tensor ModelOpt FP8 linear using Gaudi FP8 MME."""

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = _checkpoint_weight_loader(layer, extra_weight_attrs)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        weight_dtype = torch.float8_e4m3fn if self.quant_config.is_checkpoint_fp8_serialized else params_dtype
        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=weight_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("weight_scale", weight_scale)

            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = hpu_ops.requantize_with_max_scale(
                layer.weight,
                layer.weight_scale,
                layer.logical_widths,
            )
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, "input_scale", None)
        input_2d = x.reshape(-1, x.shape[-1])
        output_shape = (*x.shape[:-1], layer.weight.shape[1])
        output = hpu_ops.apply_fp8_linear_hpu(
            input=input_2d,
            weight=layer.weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            bias=bias,
            trans_B=False,
        )
        return output.narrow(0, 0, input_2d.shape[0]).reshape(output_shape)


class HPUModelOptFp8PcPtLinearMethod(LinearMethodBase):
    """Per-channel weight/per-token activation ModelOpt FP8 linear.

    Checkpoint FP8 weights stay quantized. Each flattened BF16 token receives
    a dynamic activation scale and executes through ``hpu.fp8_gemm_v2`` with
    the checkpoint's per-output-channel inverse weight scales.
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        if not self.quant_config.is_checkpoint_fp8_serialized:
            raise ValueError("FP8_PER_CHANNEL_PER_TOKEN requires an FP8-serialized checkpoint")

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = _checkpoint_weight_loader(layer, extra_weight_attrs)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise TypeError("FP8_PER_CHANNEL_PER_TOKEN checkpoint weight must remain FP8 before HPU post-processing, "
                            f"got {layer.weight.dtype}")
        if layer.weight_scale.dtype != torch.float32 or layer.weight_scale.shape != (layer.weight.shape[0], ):
            raise ValueError("FP8_PER_CHANNEL_PER_TOKEN requires one FP32 inverse scale per output channel; "
                             f"got dtype={layer.weight_scale.dtype}, shape={tuple(layer.weight_scale.shape)}")
        layer.weight = Parameter(layer.weight.data.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data.contiguous(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = x.reshape(-1, x.shape[-1])
        output_shape = (*x.shape[:-1], layer.weight.shape[1])
        output = hpu_ops.apply_fp8_linear_hpu(
            input=input_2d,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
            trans_B=False,
        )
        return output.narrow(0, 0, input_2d.shape[0]).reshape(output_shape)


modelopt.ModelOptFp8Config = HPUModelOptFp8Config
modelopt.ModelOptFp8LinearMethod = HPUModelOptFp8LinearMethod
modelopt.ModelOptFp8PcPtLinearMethod = HPUModelOptFp8PcPtLinearMethod


# ===========================================================================
# ModelOpt MIXED_PRECISION checkpoints on Gaudi2 (Qwen4Exp FP4 export)
#
# config.json declares ``quant_algo = "MIXED_PRECISION"`` with a per-module
# ``quantized_layers`` map. Three storage kinds appear, all handled here and
# byte-decoded in ``vllm_gaudi/ops/modelopt_csf.py``:
#
# * NVFP4 routed experts (48 decoder layers, ``weight_scale_encoding: csf``)
#   and the MTP experts (declared W4A16_NVFP4, but still routed experts) ->
#   weights stay 4-bit packed in HBM, the CSF block scales are decompressed on
#   the host into plain E4M3 planes, and ``weight_scale_2`` stays the
#   per-expert F32 global scale. Consumers get
#   ``(packed [E,N,K/2] U8, scales [E,N,K/16] E4M3, scale2 [E] F32)`` from
#   ``HPUModelOptMixedPrecisionConfig.load_experts``, in the resident TPC
#   layout declared in ``vllm_gaudi/ops/nvfp4_tpc_layout.py`` (the layout the
#   Track 3 GEMV reads; ``VLLM_QWEN4EXP_TPC_LAYOUT`` selects a variant).
# * MXFP8 linears (attention, linear_attn, indexer, shared experts, vision
#   attention) -> converted at load into per-channel FP8 so the Gaudi2 MME
#   runs them; the 32-wide UE8M0 block scales are folded by an exact
#   power-of-two rescale, which keeps the weights at one byte each.
# * W4A16_NVFP4 vision MLP/merger matrices (56, small) -> dequantized to BF16
#   at load with vLLM's own emulation math.
# ===========================================================================


def _hpu_gaudi2_scale_adjustment() -> bool:
    """Whether the Gaudi2 FP8 range correction applies.

    Same switch the checkpoint loaders use (``hpu_ops.gaudi_weight_wrapper``):
    the ModelOpt values were quantized against the OCP E4M3 range and Gaudi2's
    FP8 MME range is narrower, so one side is halved and the other doubled.
    """
    if not hpu_ops.is_hpu_gaudi2:
        return False
    from vllm_gaudi.extension.runtime import get_config

    return bool(get_config().scale_adjustment)


class HPUModelOptMxFp8LinearMethod(HPUModelOptFp8PcPtLinearMethod):
    """MXFP8 checkpoint layer executed as Gaudi2 per-channel FP8.

    ``create_weights`` keeps the checkpoint layout (E4M3 values, one UE8M0 byte
    per 32 inputs). ``process_weights_after_loading`` folds the block exponents
    into one inverse scale per output channel and applies the Gaudi2 range
    correction once -- so the weight loader must not wrap it in
    ``gaudi_weight_wrapper`` a second time.
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        super().__init__(quant_config)
        self.scale_adjustment = _hpu_gaudi2_scale_adjustment()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        group_size = modelopt_csf.MXFP8_GROUP_SIZE
        if input_size_per_partition % group_size:
            raise ValueError(f"MXFP8 requires K divisible by {group_size}, "
                             f"got {input_size_per_partition}")
        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is None:
            raise ValueError("ModelOpt MXFP8 HPU linear requires a checkpoint weight loader")
        # No gaudi_weight_wrapper: the conversion below performs the same
        # arithmetic exactly once, after the block scales have been folded.
        weight_loader = hpu_ops.synced_weight_loader(weight_loader)

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition,
                                 input_size_per_partition,
                                 dtype=torch.float8_e4m3fn),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition,
                                 input_size_per_partition // group_size,
                                 dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise TypeError("MXFP8 checkpoint weight must stay E4M3 until HPU post-processing, "
                            f"got {layer.weight.dtype}")
        if layer.weight_scale.dtype != torch.uint8:
            raise TypeError("MXFP8 block scales are UE8M0 bytes, "
                            f"got {layer.weight_scale.dtype}")
        weight, weight_scale = modelopt_csf.mxfp8_to_fp8_per_channel(
            layer.weight,
            layer.weight_scale,
            gaudi2_correction=self.scale_adjustment,
        )
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        # Validates the [N] F32 contract and transposes for fp8_gemm_v2.
        super().process_weights_after_loading(layer)


class HPUModelOptW4A16NvFp4LinearMethod(LinearMethodBase):
    """W4A16_NVFP4 layer for Gaudi2: dequantized to BF16 at load.

    Only used for the small vision MLP/merger matrices (and any other
    non-expert W4A16 layer). Routed experts declared W4A16_NVFP4 (the MTP
    experts) keep their 4-bit residency and go through
    ``load_experts`` instead.
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        group_size = modelopt_csf.NVFP4_GROUP_SIZE
        if input_size_per_partition % group_size:
            raise ValueError(f"NVFP4 requires K divisible by {group_size}, "
                             f"got {input_size_per_partition}")
        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is None:
            raise ValueError("ModelOpt W4A16 HPU linear requires a checkpoint weight loader")
        weight_loader = hpu_ops.synced_weight_loader(weight_loader)

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer._hpu_dequant_dtype = params_dtype or torch.bfloat16
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition,
                                 input_size_per_partition // 2,
                                 dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition,
                                 input_size_per_partition // group_size,
                                 dtype=torch.float8_e4m3fn),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        weight_scale_2[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale_2", weight_scale_2)

    def process_weights_after_loading(self, layer: Module) -> None:
        if layer.weight.dtype != torch.uint8:
            raise TypeError("W4A16_NVFP4 checkpoint weight must stay packed U8, "
                            f"got {layer.weight.dtype}")
        dtype = getattr(layer, "_hpu_dequant_dtype", torch.bfloat16)
        weight = modelopt_csf.dequant_w4a16_nvfp4(
            layer.weight,
            layer.weight_scale,
            layer.weight_scale_2.max(),
            dtype=dtype,
        )
        layer.weight = Parameter(weight, requires_grad=False)
        # The packed planes and block scales are dead once the matrix is BF16.
        for name in ("weight_scale", "weight_scale_2"):
            if hasattr(layer, name):
                delattr(layer, name)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)


def hpu_modelopt_method_for(
    quant_config: "HPUModelOptMixedPrecisionConfig",
    kind: str,
    quant_algo: str | None = None,
) -> Optional[QuantizeMethodBase]:
    """Factory: the Gaudi2 quant method for a ``modelopt_csf`` dispatch kind."""
    if kind == modelopt_csf.KIND_MXFP8:
        return HPUModelOptMxFp8LinearMethod(quant_config.mxfp8_config)
    if kind == modelopt_csf.KIND_W4A16_BF16:
        return HPUModelOptW4A16NvFp4LinearMethod(quant_config.w4a16_nvfp4_config)
    if kind == modelopt_csf.KIND_FP8:
        if quant_algo == "FP8":
            return HPUModelOptFp8LinearMethod(quant_config.fp8_config)
        return HPUModelOptFp8PcPtLinearMethod(quant_config.fp8_config)
    raise ValueError(f"no Gaudi2 ModelOpt quant method for kind {kind!r}")


class HPUModelOptMixedPrecisionConfig(ModelOptMixedPrecisionConfig):
    """MIXED_PRECISION ModelOpt config with per-prefix Gaudi2 dispatch.

    The per-module algorithm lookup is vLLM's own
    (``ModelOptMixedPrecisionConfig._resolve_quant_algo``: direct, fused,
    child and parent-prefix strategies). Only the methods that pass the packed
    4-bit experts or execute a linear change on Gaudi2.
    """

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # Same contract as HPUModelOptFp8Config: the Gaudi2 FP8 MME path runs
        # bf16 activations for both the folded MXFP8 layers and the FP8 ones.
        return [torch.bfloat16]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, (Attention, MLAAttention)):
            if self.kv_cache_quant_method:
                return self.KVCacheMethodCls(self)
            return None
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        kind = modelopt_csf.dispatch_kind(
            prefix,
            self.quantized_layers,
            self.exclude_modules,
            self.packed_modules_mapping,
        )
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if kind == modelopt_csf.KIND_NVFP4_PACKED:
                raise ValueError(
                    f"{prefix}: NVFP4 layers keep their packed planes in HBM and are consumed by "
                    "the indexed MoE kernels, not by a dense linear. Load the experts with "
                    "HPUModelOptMixedPrecisionConfig.load_experts().")
            if kind == modelopt_csf.KIND_UNQUANTIZED:
                return UnquantizedLinearMethod()
            return hpu_modelopt_method_for(self, kind, self._resolve_quant_algo(prefix))
        if isinstance(layer, RoutedExperts):
            raise ValueError(
                f"{prefix}: routed experts of a ModelOpt MIXED_PRECISION checkpoint load through "
                "HPUModelOptMixedPrecisionConfig.load_experts() (packed U8 + plain E4M3 scales + "
                "F32 scale2), not through vLLM's FusedMoE.")
        return None

    # -- host-side loaders Track D / the MoE kernels call -------------------
    @staticmethod
    def tpc_layout() -> str:
        """Resident TPC layout the next load will use (``VLLM_QWEN4EXP_TPC_LAYOUT``)."""
        return modelopt_csf.tpc_layout_module().layout_from_env()

    def load_experts(
        self,
        source,
        prefix: str,
        num_experts: int,
        *,
        projection: str = "gate_proj",
        group_size: int = modelopt_csf.NVFP4_GROUP_SIZE,
        dtype: torch.dtype = torch.float32,
        layout: str | None = None,
    ) -> modelopt_csf.NvFp4PackedWeights:
        """``(packed [E,N,K/2] U8, scales [E,N,K/16] E4M3, scale2 [E] F32)``.

        This is the whole Track B load path for routed experts, and the only
        caller of the CSF decoder in a serve: CSF scales are decoded here, on
        the host, in memory (decision A), and the planes come back already in
        the TPC layout the Track 3 GEMV reads (decision B).
        """
        return modelopt_csf.load_experts(
            source,
            prefix,
            num_experts,
            projection=projection,
            group_size=group_size,
            dtype=dtype,
            layout=layout,
        )

    def load_expert_w13(self, source, prefix: str, num_experts: int, **kwargs):
        """``gate_proj`` + ``up_proj`` stacked on N (``[E, 2N, K/2]``)."""
        return modelopt_csf.load_expert_w13(source, prefix, num_experts, **kwargs)

    def load_vision_matrix_as_bf16(self, source, base: str, *, dtype: torch.dtype = torch.bfloat16):
        """Dequantize one vision MLP/merger W4A16_NVFP4 matrix to BF16."""
        return modelopt_csf.load_w4a16_as_bf16(source, base, dtype=dtype)

    def load_mxfp8_linear(self, source, base: str, **kwargs):
        """Load one MXFP8 matrix as ``(weight_e4m3 [N,K], weight_scale [N] f32)``."""
        if "gaudi2_correction" not in kwargs:
            kwargs["gaudi2_correction"] = _hpu_gaudi2_scale_adjustment()
        return modelopt_csf.load_mxfp8_as_fp8(source, base, **kwargs)


def _register_mixed_precision_platform_support(platform: object = None) -> None:
    """Allow the "modelopt_mixed" method name on the HPU platform.

    A ``quant_method: "modelopt"`` checkpoint with
    ``quant_algo: "MIXED_PRECISION"`` is resolved by vLLM through
    ``ModelOptMixedPrecisionConfig.override_quantization_method`` to the name
    ``"modelopt_mixed"``, and ``ModelConfig._verify_quantization`` then calls
    ``current_platform.verify_quantization("modelopt_mixed")``, which raises
    unless the name is in the platform's ``supported_quantization``.  The FP8
    variants keep the already-supported ``"modelopt"`` name, so this is the
    only name that has to be added.

    Resolved lazily (and only on HPU) so importing this module on a CPU host
    never pulls in the Habana stack.
    """
    if platform is None:
        from vllm.platforms import current_platform

        platform = current_platform
    # HpuPlatform sets device_type = "hpu" (its _enum is PlatformEnum.OOT,
    # since HPU ships as a plugin); this vLLM has no Platform.is_hpu().
    if getattr(platform, "device_type", None) != "hpu":
        return
    supported = getattr(platform, "supported_quantization", None)
    if supported is not None and "modelopt_mixed" not in supported:
        supported.append("modelopt_mixed")


_register_mixed_precision_platform_support()

modelopt.ModelOptMixedPrecisionConfig = HPUModelOptMixedPrecisionConfig
