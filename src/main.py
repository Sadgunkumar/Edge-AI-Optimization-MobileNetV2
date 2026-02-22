import torch
import torchvision.models as models
import onnx
from onnxruntime.quantization import quantify_dynamic, QuantType
import os
if not os.path.exists('models'):
    os.makedirs('models')
# 1. Load Pre-trained MobileNetV2
model = models.mobilenet_v2(weights='DEFAULT')
model.eval()

# 2. Export to ONNX (FP32)
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "models/mobilenet_v2_fp32.onnx"
torch.onnx.export(model, dummy_input, onnx_path, 
                  export_params=True, opset_version=12,
                  input_names=['input'], output_names=['output'])

print(f"FP32 Model Size: {os.path.getsize(onnx_path) / 1e6:.2f} MB")

# 3. Apply Post-Training Quantization (INT8)
quantized_path = "models/mobilenet_v2_int8.onnx"
quantify_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)

print(f"INT8 Model Size: {os.path.getsize(quantized_path) / 1e6:.2f} MB")
