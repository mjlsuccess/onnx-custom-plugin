/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "customClipPlugin.h"
#include "NvInfer.h"
#include "clipKernel.h"

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Clip plugin specific constants
namespace {
    static const char* CLIP_PLUGIN_VERSION{"1"};
    static const char* CLIP_PLUGIN_NAME{"CustomClipPlugin"};
}

// Static class fields initialization
PluginFieldCollection ClipPluginCreator::mFC{};
std::vector<PluginField> ClipPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ClipPluginCreator);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

ClipPlugin::ClipPlugin(const std::string name, float clipMin, float clipMax)
    : mLayerName(name)
    , mClipMin(clipMin)
    , mClipMax(clipMax)
{
}

ClipPlugin::ClipPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mClipMin = readFromBuffer<float>(d);
    mClipMax = readFromBuffer<float>(d);

    assert(d == (a + length));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt*  ClipPlugin::clone() const
{
    auto plugin = new ClipPlugin(mLayerName, mClipMin, mClipMax);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs ClipPlugin::getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    // Validate input arguments
    printf("ClipPlugin::getOutputDimensions nbInputs: %d\n", nbInputs);
    for (int i =0; i< inputs->nbDims; i++){
        printf("%d\n", inputs->d[i]->getConstantValue());
    }
    printf("outputIndex = %d\n", outputIndex);
    assert(nbInputs == 3);
    assert(outputIndex == 0);

    // Clipping doesn't change input dimension, so output Dims will be the same as input Dims
    return *inputs;    
}

std::string print_type(nvinfer1::DataType type){
    if (type == DataType::kFLOAT){
        return std::string("kFLOAT");
    }
    else if (type == DataType::kHALF){
        return std::string("kHALF");
    }
    else if (type == DataType::kINT8){
        return std::string("kINT8");
    }
    else if (type == DataType::kINT32){
        return std::string("kINT32");
    }
    else if (type == DataType::kBOOL){
        return std::string("kBOOL");
    } 
    else{
        return std::string("NoneType");
    }           
}

std::string print_format(nvinfer1::TensorFormat format){
    if (format == TensorFormat::kLINEAR){
        return std::string("kLINEAR");
    }
    else if (format == TensorFormat::kNCHW){
        return std::string("kNCHW");
    }
    else if (format == TensorFormat::kCHW2){
        return std::string("kCHW2");
    }
    else if (format == TensorFormat::kNC2HW2){
        return std::string("kNC2HW2");
    }
    else if (format == TensorFormat::kHWC8){
        return std::string("kHWC8");
    }
    else if (format == TensorFormat::kNHWC8){
        return std::string("kNHWC8");
    }   
    else if (format == TensorFormat::kCHW4){
        return std::string("kCHW4");
    } 
    else if (format == TensorFormat::kCHW16){
        return std::string("kCHW16");
    } 
    else if (format == TensorFormat::kCHW32){
        return std::string("kCHW32");
    } 
    else {
        return std::string("NoneFormat");
    }                     
}

// 记得实现mVolume
bool ClipPlugin::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) 
{
    // 0 - input
    printf("pos: %d, type: %s, format: %s\n", pos, print_type(inOut[pos].type).c_str(), print_format(inOut[pos].format).c_str());

    // const nvinfer1::PluginTensorDesc& input = inOut[0];    
    // if (pos == 0){
    //     return (input.type == DataType::kFLOAT) && (input.format == TensorFormat::kLINEAR);
    // }
    // if (pos == 1){
    //     const nvinfer1::PluginTensorDesc& output = inOut[1];
    //     return(input.type == output.type) && (output.format == TensorFormat::kLINEAR);
    // }
    // return false;
    return (inOut[pos].type == DataType::kFLOAT);
}

void ClipPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    printf("ClipPlugin configurePlugin\n");
    // assert(mType == in[0].desc.type);    
}   

size_t ClipPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    return 0;
}

int ClipPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    size_t volume = 1;
    for (int i = 0; i < inputDesc->dims.nbDims; i++) {
        volume *= inputDesc->dims.d[i];
    }   
    mInputVolume = volume;
    // Launch CUDA kernel wrapper and save its return value
    status = clipInference(stream, mInputVolume, mClipMin, mClipMax, inputs[0], output);

    return status;    
}

// IPluginV2Ext Methods
nvinfer1::DataType ClipPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}


// IPluginV2 Methods
const char* ClipPlugin::getPluginType() const
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPlugin::getPluginVersion() const
{
    return CLIP_PLUGIN_VERSION;
}

int ClipPlugin::getNbOutputs() const
{
    return 1;
}


int ClipPlugin::initialize()
{
    return 0;
}


size_t ClipPlugin::getSerializationSize() const
{
    return 2 * sizeof(float);
}

void ClipPlugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mClipMin);
    writeToBuffer(d, mClipMax);

    assert(d == a + getSerializationSize());
}


void ClipPlugin::terminate() {}

void ClipPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void ClipPlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* ClipPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

ClipPluginCreator::ClipPluginCreator()
{
    // Describe ClipPlugin's required PluginField arguments
    mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ClipPluginCreator::getPluginName() const
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPluginCreator::getPluginVersion() const
{
    return CLIP_PLUGIN_VERSION;
}

const PluginFieldCollection* ClipPluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2* ClipPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float clipMin, clipMax;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "clipMin") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMin = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "clipMax") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMax = *(static_cast<const float*>(fields[i].data));
        }
    }
    return new ClipPlugin(name, clipMin, clipMax);
}

nvinfer1::IPluginV2* ClipPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    return new ClipPlugin(name, serialData, serialLength);
}

void ClipPluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* ClipPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
