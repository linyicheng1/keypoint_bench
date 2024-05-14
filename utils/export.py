import torch
import openvino as ov
import torch


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
                      output_names=['output'])
    ov_model = ov.convert_model(name+".onnx", example_input=dummy_input)
    ov.save_model(ov_model, name+'.xml')




