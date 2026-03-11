# TODO: Not tested since refactor
# Requirements listed in requirements/tflite.txt

if __name__ == '__main__':
    import os
    from os import path

    import ai_edge_torch

    from tools.export_utils import prepare_export_model, get_dummy_input

    model_name = "parseq"
    export_mode = 'dynamo'  # Same compatibility issues as ONNX dynamo export
    output_dir = "exported/tflite"
    os.makedirs(output_dir, exist_ok=True)

    model = prepare_export_model(model_name, export_mode)
    image = get_dummy_input()

    # Test model forward pass for debugging purposes
    # model(image)

    tf_model_dir = path.join(output_dir, model_name)
    edge_model = ai_edge_torch.convert(model, (image,), strict_export=True, _saved_model_dir=tf_model_dir)

    output_path = path.join(output_dir, f"{model_name}.tflite")
    edge_model.export(output_path)
