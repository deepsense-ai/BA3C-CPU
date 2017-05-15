/*
 * conv_ops_mkl_cache.h
 *
 *  Author: Tomasz Grel (tomasz.grel@codilime.com)
 */

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_MKL_CACHE_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_MKL_CACHE_H_

#include "include/mkl.h"
#include "include/mkl_dnn.h"

#include "tensorflow/core/kernels/mkl_convolution.h"
#include "tensorflow/core/kernels/conv_params.h"

#include <unordered_set>

namespace tensorflow {

class ConvOpsMklCache {
public:
  ~ConvOpsMklCache();

  MklConvolution& get(const conv_params_t& param);

  ConvOpsMklCache();
private:

  std::unordered_map<conv_params_t, MklConvolution*> cache_;
};

}
#endif /* TENSORFLOW_CORE_KERNELS_CONV_OPS_MKL_CACHE_H_ */

