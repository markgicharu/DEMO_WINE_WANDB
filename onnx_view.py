import onnx
import onnxruntime

model = onnx.load("models/home_state_test.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))