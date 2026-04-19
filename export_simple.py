import torch
import torch.nn as nn
import onnx

# Load the model directly
model_path = 'runs/detect/train5/weights/best.pt'

# Load with weights_only=False to avoid security restrictions
model = torch.load(model_path, map_location='cpu', weights_only=False)

# Check if it's a YOLO model
if isinstance(model, dict) and 'model' in model:
    model = model['model']

# Convert model to float32
model = model.float()
model.eval()

# Create dummy input
batch_size = 1
channels = 3
height = 640
width = 640
dummy_input = torch.randn(batch_size, channels, height, width).float()

# Export to ONNX
output_path = 'rebar_count.onnx'
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    verbose=False,
    opset_version=11,
    input_names=['images'],
    output_names=['output']
)

print(f"ONNX model exported to {output_path}")
