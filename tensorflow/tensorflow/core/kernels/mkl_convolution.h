/*
 * Author: Tomasz Grel (tomasz.grel@codilime.com)
 */

#ifndef MKL_CONVOLUTION_H
#define MKL_CONVOLUTION_H

#include <omp.h>

#include "conv_params.h"
#include "include/mkl.h"
#include "include/mkl_dnn.h"
#include <mutex>

namespace tensorflow {

class MklConvolution {
public:

  MklConvolution(const conv_params_t& params);
  ~MklConvolution();

	// todo remove these from interface
//  size_t* inputSize() { return _inputSize; }
//  size_t* outputSize() { return _outputSize; }
//  size_t* filterSize() { return _filterSize; }
//  int* offset() { return _offset; }
//  size_t* convolutionStride() { return _convolutionStride; }
//
//  const conv_params_t& params() {return params_;}
//  const dnnPrimitive_t& forwardPrimitive() {return forward_;}
//  const dnnPrimitive_t& backwardFilterPrimitive() {return backward_filter_;}
//  const dnnPrimitive_t& backwardDataPrimitive() {return backward_data_;}
  // until here

  void executeForward(const Tensor& input, const Tensor& filter, Tensor* output);
  void executeBackwardsFilter(const Tensor& input, const Tensor& out_backprop, Tensor* filter_backprop);
  void executeBackwardsData(const Tensor& filter, const Tensor& output, Tensor* input);

private:
  std::mutex lock_;
  conv_params_t params_;

  dnnPrimitive_t forward_;
  dnnPrimitive_t backward_filter_;
  dnnPrimitive_t backward_data_;

  dnnLayout_t layout_forward_src_;
  dnnLayout_t layout_forward_dst_;
  dnnLayout_t layout_forward_filter_;

  dnnLayout_t layout_backward_data_src_;
  dnnLayout_t layout_backward_data_dst_;
  dnnLayout_t layout_backward_data_filter_;

  dnnLayout_t layout_backward_filter_src_;
  dnnLayout_t layout_backward_filter_dst_;
  dnnLayout_t layout_backward_filter_filter_;

  static const int dimension = 4;

  size_t _inputSize[dimension];
  size_t _outputSize[dimension];
  size_t _filterSize[dimension];

  static const int image_dim = 2;
  int _offset[image_dim];
  size_t _convolutionStride[image_dim];


  void computeMklParams();
  void createMklPrimitives();
  void createLayouts();

  void fillInputSize(size_t* inputSize);

  void fillOutputSize(size_t* outputSize, const size_t* inputSize,
    const size_t* convolutionStride, bool padding);

  void fillFilterSize(size_t* filterSize);

  void fillOffset(const size_t* inputSize, const size_t* outputSize,
  		const size_t* filterSize, const size_t* convolutionStride,
			bool padding, int* offset);

  void fillConvolutionStride(size_t* convolutionStride);

};
} // namespace tensorflow
#endif
