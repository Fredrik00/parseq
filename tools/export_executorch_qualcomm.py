# Requirements listed in requirements/executorch.txt

import os

# TODO: Necessary for executorch?
# PyTorch export does not support fused attention as of version 2.0
os.environ['TIMM_FUSED_ATTN'] = '0'


if __name__ == '__main__':
    from tools.qualcomm_utils import build_executorch_binary
    from tools.export_utils import get_dummy_input, prepare_export_model

    model_name = "parseq"
    export_mode = 'executorch'

    backends = [
        "vulkan",
        "xnnpack"
    ]

    model = prepare_export_model(model_name, export_mode)
    image = get_dummy_input()

    # Test model forward pass for debugging purposes
    # model(image)

    inputs = (image,)
    build_executorch_binary(
        model,
        inputs,
        "SM8650",
        "parseq_qualcomm.pte",
        [inputs],
        skip_node_op_set={
            "aten.linear.default",  # Causes NPE (weight_tensor is None when calling define_tensor)
            "aten.ge.Scalar",  # Causes KeyError (not in node_visitors during is_node_supported method)
        },
        # quant_dtype=QuantDtype.use_8a8w,  # (RuntimeError: The size of tensor a (8) must match the size of tensor b (192) at non-singleton dimension 2)
    )
