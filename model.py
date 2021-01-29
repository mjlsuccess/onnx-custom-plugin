import torch
from torch.autograd import Function
from collections import OrderedDict
import onnx_graphsurgeon as gs
import onnx


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.relu = torch.nn.ReLU6(inplace=True)
        self.conv2d = torch.nn.Conv2d(32, 16, 1)

    def add(self, points):
        return points+10.0

    def forward(self, points):
        points = self.relu(points)
        points = self.add(points)
        re = self.conv2d(points)
        return re


def export_onnx(onnx_filename):
    points = torch.full((16, 32, 300, 300), 1.5, dtype=torch.float32).cuda()
    inputs = (points, )

    model = CustomModel().cuda()

    torch.onnx.export(model, inputs, onnx_filename, opset_version=11)

    print("The output of raw network: ", torch.mean(model(*inputs)).detach().cpu().numpy())


def print_onnx_model(onnx_filename):
    # Load the ONNX model
    import onnx
    model = onnx.load(onnx_filename)
    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))  


#ClipPlugin for relu6 has been a built-in op in tensorrt, and we map the relu6 to CustomClipPlugin only for demo.
def replace_relu6_with_customClipPlugin(old_onnx_filename, new_onnx_filename):  
    model = onnx.load(old_onnx_filename)

    graph = gs.import_onnx(model)
    tensors = graph.tensors()

    graph.cleanup()

    #map Clip op to CustomClipPlugin
    for node in graph.nodes:
        if "Clip" in node.name :
            node.name = "Clip_plugin"
            node.op = "CustomClipPlugin" # keep the same with CLIP_PLUGIN_NAME in customClipPlugin.cpp
            node.attrs = OrderedDict({"clipMin":0.0, "clipMax":6.0}) 

    onnx.save(gs.export_onnx(graph), new_onnx_filename)    


if __name__ == '__main__':
    onnx_filename = "models/test_model.onnx"
    export_onnx(onnx_filename)
    # print_onnx_model(onnx_filename)

    new_onnx_filename = "models/test_model_mod.onnx"
    replace_relu6_with_customClipPlugin(onnx_filename, new_onnx_filename)


