import onnxruntime as ort

session = ort.InferenceSession("models/yolov8n.onnx")

print("ONNX loaded successfully!")
