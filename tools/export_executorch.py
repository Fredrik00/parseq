# Requirements listed in requirements/executorch.txt

if __name__ == '__main__':
    import os
    from os import path

    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    from torch.export import export

    from tools.export_utils import get_dummy_input, prepare_export_model

    model_name = "parseq"
    export_mode = 'executorch'
    output_dir = "exported/executorch"
    os.makedirs(output_dir, exist_ok=True)

    backends = [
        "vulkan",
        "xnnpack"
    ]

    model = prepare_export_model(model_name, export_mode)
    image = get_dummy_input()

    # Test model forward pass for debugging purposes
    # model(image)

    inputs = (image,)
    exported_program = export(model, inputs)  # TODO: Add dynamic batch size

    partitioner = []
    if "vulkan" in backends:
        partitioner.append(VulkanPartitioner())
    if "xnnpack" in backends:
        partitioner.append(XnnpackPartitioner())

    executorch_program = to_edge_transform_and_lower(exported_program, partitioner=partitioner).to_executorch()

    output_path = path.join(output_dir, f"{model_name}_{'_'.join(backends)}.pte")
    with open(output_path, "wb") as file:
        file.write(executorch_program.buffer)
