# Adding A Custom Layer To Your Pytorch Network In TensorRT In Python

## Description 
This sample, `onnx_custom_plugin`, demonstrates how to use plugins written in C++ with the TensorRT Python bindings and onnx Parser. This sample is based on office sample `uff_custom_plugin`.


## How does this sample work?

This sample implements a clip layer (as a CUDA kernel), wraps the implementation in a TensorRT plugin (with a corresponding plugin creator) and then generates a shared library module containing its code. The user then dynamically loads this library in Python, which causes the plugin to be registered in TensorRT's PluginRegistry and makes it available to the onnx parser.

This sample includes:
`plugin/`
This directory contains files for the Clip layer plugin.

`clipKernel.cu`
A CUDA kernel that clips input.

`clipKernel.h`
The header exposing the CUDA kernel to C++ code.

`customClipPlugin.cpp`
A custom TensorRT plugin implementation, which uses the CUDA kernel internally.

`customClipPlugin.h`
The ClipPlugin headers.

`model.py`
This script generates a model with ReLU6 layer, then maps the ReLu6 in onnx to the CustomClipPlugin.

`sample.py`
This script converts the onnx to trt model, then does inference on a sample data.


## Prerequisites
`pytorch==1.7.1 tenorrt==7.1.3.4 onnx_graphsurgeon==0.2.3 onnx==1.6.0 pycuda==2020.1`

## Running the sample

1.  Build the plugin and its corresponding Python bindings.
	```
	mkdir build
	cmake ..
	```

	**Note:** If any of the dependencies are not installed in their default locations, you can manually specify them. For example:
    ```
    cmake .. -DPYBIND11_DIR=/path/to/pybind11/
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-x.x/bin/nvcc  (Or adding path/to/nvcc into $PATH)
    -DPYTHON3_INC_DIR=/usr/include/python3.6/
    -DTRT_LIB=/path/to/tensorrt/lib/
    -DTRT_INCLUDE=/path/to/tensorrt/include/
    ```
	
	`cmake ..` displays a complete list of configurable variables. If a variable is set to `VARIABLE_NAME-NOTFOUND`, then you’ll need to specify it manually or set the variable it is derived from correctly.

2.  Build the plugin.   
	```
	make 
	```

3.  Run the sample to generate the model:
    ```
	python3 model.py
	```
	It outputs:
	`The output of raw network:  -0.23130444`.
	
	You can visualize the onnx models with [Netron](https://github.com/lutzroeder/netron).

4.  Run inference using TensorRT with the custom clip plugin implementation:
    ```
	python3 sample.py
	```
	It outputs:
	`The output of TRT:  -0.23130527`.


# Additional resources

The following resources provide a deeper understanding about getting started with TensorRT using Python:

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

