/*
 * conv_ops_mkl_cache.cpp
 *
 *  Author: Tomasz Grel (tomasz.grel@codilime.com)
 */

#include "tensorflow/core/kernels/mkl_convolution.h"
#include "tensorflow/core/kernels/conv_ops_mkl_cache.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

ConvOpsMklCache::ConvOpsMklCache() {}

ConvOpsMklCache::~ConvOpsMklCache() {
  for (auto& key_value_pair : cache_) {
    delete key_value_pair.second;
  }
}

MklConvolution& ConvOpsMklCache::get(const conv_params_t& param) {
  auto it = cache_.find(param);
  if (it != cache_.end()) {
    LOG(DEBUG) << "Returning the MklConvolution from cache";
    return *it->second;
  }

  LOG(DEBUG) << "Creating the MklConvolution";
  MklConvolution* new_convolution = new MklConvolution(param);
  cache_[param] = new_convolution;
  return *new_convolution;
}
}
