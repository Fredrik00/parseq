# Requirements listed in requirements/executorch.txt

# Make sure to source $QNN_SDK_ROOT/bin/envsetup.sh before running this script
if __name__ == '__main__':
    from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
    # from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

    from tools.export_utils import get_dummy_input, prepare_export_model
    from tools.qualcomm_utils import build_executorch_binary

    model_name = "parseq"
    export_mode = "qualcomm"

    model = prepare_export_model(model_name, export_mode)
    image = get_dummy_input()

    # Test model forward pass for debugging purposes
    # model(image)

    model = convert_linear_to_conv2d(model)

    inputs = (image,)
    build_executorch_binary(
        model,
        inputs,
        "SM8650",
        "parseq_qualcomm.pte",
        [inputs],
        skip_node_op_set={
            "aten.full.default", # avoid zero input for QNN Graph. If other operations can be fully delegated, this skip may be removed.
            "aten.where.self" # prepare failed from MultiheadAttention.
        },
        skip_node_id_set={
            "aten_view_copy_default_136" # prepare failed in QNN 2.37, but successful in QNN 2.40
        },
        # quant_dtype=QuantDtype.use_8a8w,  # (RuntimeError: The size of tensor a (8) must match the size of tensor b (192) at non-singleton dimension 2)
    )
