# Requirements listed in requirements/onnx.txt

if __name__ == '__main__':
    import os
    from os import path

    import torch
    from torch.export import Dim

    from tools.export_utils import get_dummy_input, prepare_export_model

    model_name = "parseq"
    export_mode = 'dynamo'
    dynamic_batch_size = False
    output_dir = "exported/onnx"
    os.makedirs(output_dir, exist_ok=True)

    model = prepare_export_model(model_name, export_mode)
    image = get_dummy_input()

    # Test model forward pass for debugging purposes
    # model(image)

    export_opts = {}
    if dynamic_batch_size:
        export_opts["args"] = image.repeat(2, 1, 1, 1)
        export_opts["dynamic_shapes"] = {"images": (Dim.DYNAMIC, Dim.STATIC, Dim.STATIC, Dim.STATIC)}
    else:
        export_opts["args"] = image

    # Save model manually after export, otherwise we get a separate data file
    onnx_model = torch.onnx.export(
        model,
        input_names=['images'],
        output_names=['outputs'],
        dynamo=True,
        optimize=True,
        verbose=True,  # Includes metadata used during quantization
        **export_opts
    )

    output_path = path.join(output_dir, f"{model_name}.onnx")
    onnx_model.save(output_path)
