/*
 * Author: Tomasz Grel (tomasz.grel@codilime.com)
 */


#ifndef CONV_OPS_MKL_PARAM_H
#define CONV_OPS_MKL_PARAM_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/util/tensor_format.h"
#include <math.h>

#include "include/mkl.h"
#include "include/mkl_dnn.h"

#include <ostream>

#define DEFAULT_STRIDES(strides, size, count) do { \
    ((strides)[0]) = 1; \
    for (size_t i = 1; i < count; ++i) \
        ((strides)[i]) = ((strides)[i-1])*((size)[i-1]); \
} while(0);


#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        LOG(WARNING) << "ERROR: " << __FILE__ << ":" << __LINE__ << " " << err;\
        return -1; \
    } \
} while(0);

namespace tensorflow {
  class conv_params_t;
}

namespace std {
  template <>
  class hash<tensorflow::conv_params_t> {
  public:
      std::size_t operator()(const tensorflow::conv_params_t& p) const;
  };
}

namespace tensorflow {


enum convolution_direction_t {
     FORWARD = 2,
     BACKWARD_FILTER = 3,
     BACKWARD_DATA = 4
};

class conv_params_t {
  friend class std::hash<tensorflow::conv_params_t>;

public:
  int groups;
  int minibatch;
  int w;
  int h;
  int ic;
  int oc;
  int kw;
  int kh;
  int c_stride;
  int iters;
  bool padding; // true means SAME padding
  TensorFormat data_format;

  bool operator==(const conv_params_t& rhs) const;
};

void filters_convert_MKL(Tensor& tf_filters, float* mkl_array,
              const conv_params_t &param, bool direction_mkl_to_tf,
              dnnLayout_t mkl_layout);

void src_convert_MKL(Tensor& tf_src, float* mkl_array,
              const conv_params_t &param, bool direction_mkl_to_tf,
              TensorFormat format, dnnLayout_t mkl_layout);

void dst_convert_MKL(Tensor& tf_dst, float* mkl_array,
          size_t* output_size, bool direction_mkl_to_tf,
          TensorFormat data_format_, dnnLayout_t mkl_layout);

Eigen::array<int, 4> get_shuffle(TensorFormat tf_format, TensorFormat mkl_format);

std::ostream& operator<< (std::ostream& os, const conv_params_t& p);

} //namespace tensorflow

#endif
