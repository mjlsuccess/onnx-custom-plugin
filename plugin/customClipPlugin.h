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

#ifndef CUSTOM_CLIP_PLUGIN_H
#define CUSTOM_CLIP_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>


using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class ClipPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    ClipPlugin(const std::string name, float clipMin, float clipMax);

    ClipPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make ClipPlugin without arguments, so we delete default constructor.
    ClipPlugin() = delete;


    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;
    
    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;


     // IPluginV2 Methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    float mClipMin, mClipMax;
    size_t mInputVolume;
    std::string mNamespace;

protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;    
};

class ClipPluginCreator : public nvinfer1::IPluginCreator
{
public:
    ClipPluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    
    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif
