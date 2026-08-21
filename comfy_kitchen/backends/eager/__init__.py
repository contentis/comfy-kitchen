__all__ = [
    "adaln",
    "na3d",
    "rms_adaln",
    "apply_rope",
    "apply_rope_",
    "apply_rope1",
    "apply_rope1_",
    "apply_rope_split_half",
    "apply_rope_split_half_",
    "apply_rope_split_half1",
    "apply_rope_split_half1_",
    "rms_rope",
    "rms_rope_",
    "rms_rope1",
    "rms_rope1_",
    "rms_rope_split_half",
    "rms_rope_split_half_",
    "rms_rope_split_half1",
    "rms_rope_split_half1_",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_dtype",
    "dequantize_int8_embedding",
    "dequantize_convrot_w4a4_weight",
    "dequantize_w4a8_int8_weight",
    "gemv_awq_w4a16",
    "convrot_w4a4_linear",
    "prepare_int4_weight_for_int8_linear",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "quantize_convrot_w4a4_weight",
    "quantize_w4a8_int8_weight",
    "quantize_int8_rowwise",
    "quantize_int8_convrot_weight",
    "rotate_int8_convrot_weight",
    "quantize_and_rotate_rowwise",
    "quantize_int8_tensorwise",
    "quantize_svdquant_w4a4",
    "scaled_mm_mxfp8",
    "scaled_mm_nvfp4",
    "scaled_mm_svdquant_w4a4",
    "stochastic_rounding_fp8",
    "int8_linear",
    "w4a8_int8_linear",
]

from comfy_kitchen.registry import registry

from .adaln import adaln, rms_adaln
from .awq import gemv_awq_w4a16
from .convrot_w4a4 import (
    convrot_w4a4_linear,
    dequantize_convrot_w4a4_weight,
    prepare_int4_weight_for_int8_linear,
    quantize_convrot_w4a4_weight,
)
from .na import na3d
from .quantization import (
    dequantize_int8_convrot_weight,
    dequantize_int8_convrot_weight_dtype,
    dequantize_int8_embedding,
    dequantize_int8_simple,
    dequantize_int8_simple_dtype,
    dequantize_mxfp8,
    dequantize_nvfp4,
    dequantize_per_tensor_fp8,
    int8_linear,
    quantize_and_rotate_rowwise,
    quantize_int8_convrot_weight,
    quantize_int8_rowwise,
    quantize_int8_tensorwise,
    quantize_mxfp8,
    quantize_nvfp4,
    quantize_per_tensor_fp8,
    rotate_int8_convrot_weight,
    scaled_mm_mxfp8,
    scaled_mm_nvfp4,
    stochastic_rounding_fp8,
)
from .rope import (
    apply_rope,
    apply_rope1,
    apply_rope1_,
    apply_rope_,
    apply_rope_split_half,
    apply_rope_split_half1,
    apply_rope_split_half1_,
    apply_rope_split_half_,
    rms_rope,
    rms_rope1,
    rms_rope1_,
    rms_rope_,
    rms_rope_split_half,
    rms_rope_split_half1,
    rms_rope_split_half1_,
    rms_rope_split_half_,
)
from .svdquant import quantize_svdquant_w4a4, scaled_mm_svdquant_w4a4
from .w4a8_int8 import (
    dequantize_w4a8_int8_weight,
    quantize_w4a8_int8_weight,
    w4a8_int8_linear,
)


def _register():
    registry.register(
        name="eager",
        module=__import__(__name__, fromlist=__all__),
    )


_register()
