/* 

Author: Tomasz Grel (tomasz.grel@codilime.com)

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_grad_ops.h"

#include "tensorflow/core/kernels/debug_utils.h"

#include "include/mkl.h"
#include "include/mkl_dnn.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/kernels/conv_params.h"
#include "tensorflow/core/kernels/mkl_convolution.h"

#include "tensorflow/core/kernels/conv_ops_mkl_cache.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T>
class Conv2DBackpropInputMklOp : public OpKernel {
 public:
  explicit Conv2DBackpropInputMklOp(OpKernelConstruction* context)
      : OpKernel(context)
      , conv_(nullptr) {
    std::string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }
  ~Conv2DBackpropInputMklOp() {
	if (conv_ != nullptr) {
      dnnDelete_F32(conv_);
	}
  }

  void Compute(OpKernelContext* context) override {

    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DFastBackpropInput", input_shape,
                                filter.shape(), out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    conv_params_t params;
    params.data_format = data_format_;
    params.h = dims.rows.input_size;
    params.w = dims.cols.input_size;
    params.c_stride = dims.cols.stride;// currently only equal-sized strides in both directions, can be changed
	params.ic = dims.in_depth;
	params.minibatch = dims.batch_size;
	params.oc = dims.out_depth;
    params.kh = dims.rows.filter_size;
    params.kw = dims.cols.filter_size;
    params.padding = padding_ == SAME;

    MklConvolution& convolution = cache_.get(params);
	convolution.executeBackwardsData(filter, out_backprop, in_backprop);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
  dnnPrimitive_t conv_;
  ConvOpsMklCache cache_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DBackpropInputMklOp);
};


#define REGISTER_CPU_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_CPU)                            \
                              .Label("MKL")                               \
                              .TypeConstraint<T>("T"),                       \
                          Conv2DBackpropInputMklOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

}  // namespace tensorflow
