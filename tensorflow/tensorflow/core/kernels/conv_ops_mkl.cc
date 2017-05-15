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

#include "tensorflow/core/kernels/conv_ops.h"
#include <string.h>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>

#include <omp.h>
#include <memory>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/deep_conv2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#include "tensorflow/core/kernels/conv_params.h"
#include "tensorflow/core/kernels/mkl_convolution.h"

#include "tensorflow/core/kernels/conv_ops_mkl_cache.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class Conv2DMklOp : public BinaryOp<T> {
 public:
  explicit Conv2DMklOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    std::string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

  }

  void Compute(OpKernelContext* context) override {
      // Input tensor is of the following dimensions:
      // [ batch, in_rows, in_cols, in_depth ]

      // Input tensor is of the following dimensions:
      // [ batch, in_rows, in_cols, in_depth ]

      const Tensor& input = context->input(0);

      // Input filter is of the following dimensions:
      // [ filter_rows, filter_cols, in_depth, out_depth]
      const Tensor& filter = context->input(1);

      // For 2D convolution, there should be 4 dimensions.
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
      OP_REQUIRES(context, filter.dims() == 4,
                  errors::InvalidArgument("filter must be 4-dimensional: ",
                                          filter.shape().DebugString()));

      for (int i = 0; i < 3; i++) {
        OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                             std::numeric_limits<int>::max()),
                    errors::InvalidArgument("filter too large"));
      }

      // The last dimension for input is in_depth. It must be the same as the
      // filter's in_depth.
      const int64 in_depth = GetTensorDim(input, data_format_, 'C');
      OP_REQUIRES(
          context, in_depth == filter.dim_size(2),
          errors::InvalidArgument("input and filter must have the same depth: ",
                                  in_depth, " vs ", filter.dim_size(2)));

      // The last dimension for filter is out_depth.
      const int out_depth = static_cast<int>(filter.dim_size(3));

      // The second dimension for input is rows/height.
      // The first dimension for filter is rows/height.
      const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
      OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                           std::numeric_limits<int>::max()),
                  errors::InvalidArgument("Input rows too large"));
      const int input_rows = static_cast<int>(input_rows_raw);
      const int filter_rows = static_cast<int>(filter.dim_size(0));

      // The third dimension for input is columns/width.
      // The second dimension for filter is columns/width.
      const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
      OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                           std::numeric_limits<int>::max()),
                  errors::InvalidArgument("Input cols too large"));
      const int input_cols = static_cast<int>(input_cols_raw);
      const int filter_cols = static_cast<int>(filter.dim_size(1));

      // The first dimension for input is batch.
      const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
      OP_REQUIRES(context,
                  FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                  errors::InvalidArgument("batch is too large"));
      const int batch = static_cast<int>(batch_raw);

      // For now we take the stride from the second and third dimensions only (we
      // do not support striding on the batch or depth dimension).
      const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
      const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

      int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                           padding_, &out_rows, &pad_rows));
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                           padding_, &out_cols, &pad_cols));
      Tensor* output = nullptr;
      TensorShape out_shape =
      ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

      // If there is nothing to compute, return.
      if (out_shape.num_elements() == 0) {
        return;
      }

      conv_params_t params;

      params.h = input_rows;
      params.w = input_cols;
      params.c_stride = stride_cols;
      params.groups = 1;
	  params.ic = in_depth;
	  params.iters = 1;
	  params.minibatch = batch;
	  params.oc = out_depth;
      params.padding = padding_ == SAME;
      params.kh = filter_rows;
      params.kw = filter_cols;
      params.data_format = data_format_;

  	  MklConvolution& convolution = cache_.get(params);
      convolution.executeForward(input, filter, output);
  }

 private:

  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;
  ConvOpsMklCache cache_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DMklOp);
};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv2D") \
	  .Device(DEVICE_CPU)\
	  .Label("MKL")\
	  .TypeConstraint<T>("T"), \
      Conv2DMklOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);

  Eigen::array<int, 4> get_shuffle(TensorFormat tf_format, TensorFormat mkl_format) {
    Eigen::array<int, 4> shuffle;

    if (tf_format != mkl_format) {
      std::cout << "returning shuffle. conversion needed\n";
      shuffle = {0, 3, 2, 1};
    } else {
      std::cout << "returning dummy shuffle. only copying\n";
      shuffle = {0, 2, 1, 3};
    }
    return shuffle;
  }
}  // namespace tensorflow

