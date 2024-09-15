import torch
import openvino as ov
import torch
import tensorrt as trt

def export_model(model, name):
    device = torch.device("cpu")
    model.to(device)
    dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True).to(device)
    torch.onnx.export(model,
                      dummy_input,
                      name+".onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['score', 'descriptor'])
    ov_model = ov.convert_model(name+".onnx", example_input=dummy_input)
    ov.save_model(ov_model, name+'.xml')

    # export to tensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(name+".onnx", 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise Exception("Failed to parse the ONNX file")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    serialized_engine = builder.build_serialized_network(network, config)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    trt_engine_path = name + ".trt"
    with open(trt_engine_path, "wb") as f:
        f.write(engine.serialize())









