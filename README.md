# Edge-AI-Optimization-MobileNetV2
Optimizing MobileNetV2 for Edge deployment using PyTorch, ONNX, and INT8 Quantization.

# Edge AI Optimization: MobileNetV2 for Constrained Hardware

This project converts and optimizes a PyTorch MobileNetV2 model for deployment on Edge devices (like NVIDIA Jetson or Rockchip). It focuses on reducing memory footprint and latency while maintaining accuracy.

## 📊 Optimization Results
| Metric | FP32 (Base) | INT8 (Quantized) | Reduction |
| :--- | :--- | :--- | :--- |
| Model Size | 14.2 MB | 3.5 MB | **75.3%** |

## 🛠 Features
- **PyTorch to ONNX Export:** Ensures model portability.
- **Post-Training Quantization (PTQ):** Converts weights to INT8 to save memory.
- **Graph Visualization:** Verified via Netron.

### 🛠 Engineering Challenges Overcome
* **Version Compatibility:** Resolved conflicts between PyTorch 2.10's Dynamo engine and ONNX Opset versions by implementing a forced legacy export path (`dynamo=False`).
* **Shape Inference:** Debugged and fixed `InferenceError` related to dimension mismatches (1280 vs 1000) by utilizing a stable Opset 13 trace.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run optimization: `python src/main.py`
