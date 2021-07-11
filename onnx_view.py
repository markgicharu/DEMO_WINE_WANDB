import onnx
import onnxruntime

model = onnx.load("models/fashion_mnist_{t}.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))