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

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run optimization: `python src/main.py`
