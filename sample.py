from collections import OrderedDict
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import common
import ctypes
import numpy as np

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class ModelData(object):
    MODEL_PATH = "models/test_model_mod.onnx"
    PLUGIN_PATH = "build/libclipplugin.so"
    INPUT_SHAPE = (16, 32, 300, 300)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32
    
    
# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network:

        parser = trt.OnnxParser(network, TRT_LOGGER)
        builder.max_workspace_size = common.GiB(8)

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)


def load_test_case(pagelocked_buffer):   
    raw_data = np.full(ModelData.INPUT_SHAPE, 1.5).astype(trt.nptype(ModelData.DTYPE)).ravel()
    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, raw_data)


def main():

    ctypes.CDLL(ModelData.PLUGIN_PATH)
    onnx_model_file = ModelData.MODEL_PATH

    # Build a TensorRT engine.
    with build_engine_onnx(onnx_model_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            load_test_case(inputs[0].host)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            print("The output of TRT: ", np.mean(trt_outputs[0]))
            pass
        

if __name__ == '__main__':
    main()
