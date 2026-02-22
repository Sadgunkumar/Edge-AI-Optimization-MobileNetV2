import torch
import torchvision.models as models
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Create folder
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load Model
print("Loading MobileNetV2...")
model = models.mobilenet_v2(weights='DEFAULT')
model.eval()

# 2. Export to ONNX (The "Force Legacy" Method)
print("Exporting to ONNX (FP32)...")
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "models/mobilenet_v2_fp32.onnx"

# In PyTorch 2.10+, we use 'dynamo=False' to bypass the new bug-prone engine
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=13, 
    input_names=['input'], 
    output_names=['output'],
    dynamo=False # This is the "Magic Switch" to avoid the errors you saw
)

fp32_size = os.path.getsize(onnx_path) / 1e6
print(f"FP32 Model Size: {fp32_size:.2f} MB")

# 3. Quantize
print("Quantizing to INT8...")
quantized_path = "models/mobilenet_v2_int8.onnx"

quantize_dynamic(
    model_input=onnx_path, 
    model_output=quantized_path, 
    weight_type=QuantType.QInt8
)

int8_size = os.path.getsize(quantized_path) / 1e6
print(f"INT8 Model Size: {int8_size:.2f} MB")
print(f"Success! Model size reduced by {((fp32_size - int8_size) / fp32_size) * 100:.2f}%")
