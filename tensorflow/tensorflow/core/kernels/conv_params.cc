/*
 * Author: Tomasz Grel (tomasz.grel@codilime.com)
 *
 */

#include "tensorflow/core/kernels/conv_params.h"

#include "include/mkl.h"
#include "include/mkl_dnn.h"

#include "tensorflow/core/kernels/debug_utils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include <ostream>
//
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
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/util/tensor_format.h"
#include <math.h>

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA
//


#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/device_base.h"

#include <cstdlib>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

void convert_data(void* from, dnnLayout_t from_layout,
    void* to, dnnLayout_t to_layout) {

  dnnPrimitive_t conversion;
  int res = dnnConversionCreate_F32(&conversion, from_layout, to_layout);
  if (res != 0) {
    LOG(WARNING) << "Conversion create failed!";
    return;
  }

  res = dnnConversionExecute_F32(conversion, from, to);
  if (res != 0) {
    LOG(WARNING) << "Conversion create failed!";
  }

  dnnDelete_F32(conversion);
}

void raw_copy(Tensor& tensor, float* mkl_array, std::size_t size, bool direction_mkl_to_tf) {
	  char* tf_buffer = const_cast<char*>(tensor.tensor_data().data());
	  float* unsafe_tf_buffer = reinterpret_cast<float*> (tf_buffer);

	  if (direction_mkl_to_tf) {
		  std::copy(mkl_array, mkl_array + size, unsafe_tf_buffer);
	  } else {
		  std::copy(unsafe_tf_buffer, unsafe_tf_buffer + size, mkl_array);
	  }
}

void filters_convert_MKL(Tensor& tf_filters, float* mkl_array,
    const conv_params_t &param, bool direction_mkl_to_tf,
    dnnLayout_t mkl_layout) {

  size_t mem_size = param.oc * param.ic * param.kw * param.kh;

	raw_copy(tf_filters, mkl_array, mem_size, direction_mkl_to_tf);

  const int dimension = 4;
  size_t size[dimension] = {param.kw, param.kh, param.ic, param.oc};
  size_t strides_io[dimension] = {1, size[0], size[0] * size[1], size[0] * size[1] * size[2]};

  dnnLayout_t layout_io;
  int res = dnnLayoutCreate_F32(&layout_io, dimension, size, strides_io);
  if (res != 0) {
    LOG(WARNING) << "error creating layout!";
    return;
  }

  dnnPrimitive_t conversion;
  std::unique_ptr<float> temp_data(new float[mem_size]);

  Eigen::array<int, 4> shuffle({3, 2, 1, 0});
  if (direction_mkl_to_tf) {

    convert_data(mkl_array, mkl_layout, temp_data.get(), layout_io);

    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor, long int>, Eigen::Aligned>
    temp(temp_data.get(), {param.oc, param.ic, param.kw, param.kh});

    auto tf = tf_filters.shaped<float, 4>({param.kh, param.kw, param.ic, param.oc});
    tf = temp.shuffle(shuffle);
  } else {
    auto tf = tf_filters.shaped<float, 4>({param.kh, param.kw, param.ic, param.oc});

    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor, long int>, Eigen::Aligned>
    temp(temp_data.get(), {param.oc, param.ic, param.kw, param.kh});
    temp = tf.shuffle(shuffle);

    convert_data(temp_data.get(), layout_io,
        mkl_array, mkl_layout);
  }

  dnnLayoutDelete_F32(layout_io);
}

void src_convert_MKL(Tensor& tf_src, float* mkl_array,
    const conv_params_t &param, bool direction_mkl_to_tf,
    TensorFormat data_format_, dnnLayout_t mkl_layout) {

  const int dimension = 4;
  size_t inputSize[dimension] = {param.h, param.w, param.ic, param.minibatch};

  // NHWC strides
  size_t strides[dimension] = {inputSize[1] * inputSize[2], inputSize[2], 1, inputSize[0] * inputSize[1] * inputSize[2]};

  dnnLayout_t tf_layout;
  int res = dnnLayoutCreate_F32(&tf_layout, dimension, inputSize, strides);
  if (res != 0) {
    LOG(WARNING) << "MKL src layout create failed";
    return;
  }

  char* unsafe_tf_buffer = const_cast<char*>(tf_src.tensor_data().data());

  if (direction_mkl_to_tf) {
    convert_data(static_cast<void*> (mkl_array), mkl_layout,
        static_cast<void*>(unsafe_tf_buffer), tf_layout);
  } else {
    convert_data(static_cast<void*>(unsafe_tf_buffer), tf_layout,
        static_cast<void*> (mkl_array), mkl_layout);
  }

  dnnLayoutDelete_F32(tf_layout);
}

void dst_convert_MKL(Tensor& tf_dst, float* mkl_array,
    size_t* outputSize, bool direction_mkl_to_tf,
    TensorFormat data_format_, dnnLayout_t mkl_layout) {

  const int dimension = 4;
  size_t strides[dimension] = {outputSize[1] * outputSize[2], outputSize[2], 1, outputSize[0] * outputSize[1] * outputSize[2]};

  dnnLayout_t tf_layout;
  int res = dnnLayoutCreate_F32(&tf_layout, dimension, outputSize, strides);
  if (res != 0) {
    LOG(WARNING) << "MKL dst layout create failed";
    return;
  }

  char* unsafe_tensorflow_buffer = const_cast<char*>(tf_dst.tensor_data().data());

  if(direction_mkl_to_tf) {
    convert_data(static_cast<void*>(mkl_array), mkl_layout,
        static_cast<void*>(unsafe_tensorflow_buffer), tf_layout);
  } else {
    convert_data(static_cast<void*>(unsafe_tensorflow_buffer), tf_layout,
        static_cast<void*>(mkl_array), mkl_layout);
  }

  dnnLayoutDelete_F32(tf_layout);
}

bool conv_params_t::operator==(const conv_params_t& rhs) const {
	  return
           this->groups == rhs.groups
        && this->minibatch == rhs.minibatch
        && this->w == rhs.w
        && this->h == rhs.h
        && this->ic == rhs.ic
        && this->oc == rhs.oc
        && this->kw == rhs.kw
        && this->kh == rhs.kh
        && this->c_stride == rhs.c_stride
        && this->iters == rhs.iters
        && this->padding == rhs.padding
        && this->data_format == rhs.data_format;
}

std::ostream& operator<< (std::ostream& os, const conv_params_t& p) {
	  os << "batch: " << p.minibatch
	     << ", ic: " << p.ic
	     << ", w: " << p.w
	     << ", h: " << p.h
	     << ", oc: " << p.oc
	     << ", kh: " << p.kh
	     << ", kw: " << p.kw;
	  return os;
}
} //namespace tensorflow

namespace std {
  std::size_t hash<tensorflow::conv_params_t>::operator()(const tensorflow::conv_params_t& p) const {
	  //TODO implement the hash function
	  std::size_t hash =
		p.groups +
		p.minibatch +
		p.w +
		p.h +
		p.ic +
		p.oc +
		p.kw +
		p.kh +
		p.c_stride +
		p.iters +
		p.padding +
		p.data_format;

	  return hash;
  }
}
