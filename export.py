import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck

from ultralytics.nn import tasks
original_torch_safe_load = tasks.torch_safe_load

def patched_torch_safe_load(file):
    return torch.load(file, map_location='cpu', weights_only=False), file

tasks.torch_safe_load = patched_torch_safe_load

from ultralytics import YOLO
import onnx

def export_to_onnx(model_path='runs/detect/train5/weights/best.pt', output_path='rebar_count.onnx'):
    model = YOLO(model_path)
    
    im = torch.zeros(1, 3, 640, 640).float()
    
    torch.onnx.export(
        model.model,
        im,
        output_path,
        verbose=False,
        opset_version=11,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None,
    )
    
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX model exported to {output_path}")
    return output_path

if __name__ == '__main__':
    export_to_onnx()
