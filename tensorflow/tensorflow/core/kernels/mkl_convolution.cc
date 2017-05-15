/*
 * Author: Tomasz Grel (tomasz.grel@codilime.com)
 */

#include "tensorflow/core/kernels/mkl_convolution.h"
#include "tensorflow/core/kernels/debug_utils.h"
#include <sstream>
#include <thread>

namespace tensorflow {

MklConvolution::MklConvolution(const conv_params_t& params)
      : params_(params) {
  computeMklParams();
  createMklPrimitives();
  createLayouts();
}

MklConvolution::~MklConvolution() {
  std::lock_guard<std::mutex> guard(lock_);

  dnnDelete_F32(forward_);
  dnnDelete_F32(backward_filter_);
  dnnDelete_F32(backward_data_);

  dnnLayoutDelete_F32(layout_forward_dst_);
  dnnLayoutDelete_F32(layout_forward_src_);
  dnnLayoutDelete_F32(layout_forward_filter_);

  dnnLayoutDelete_F32(layout_backward_filter_filter_);
  dnnLayoutDelete_F32(layout_backward_filter_dst_);
  dnnLayoutDelete_F32(layout_backward_filter_src_);

  dnnLayoutDelete_F32(layout_backward_data_filter_);
  dnnLayoutDelete_F32(layout_backward_data_dst_);
  dnnLayoutDelete_F32(layout_backward_data_src_);
}

void MklConvolution::executeForward(const Tensor& input, const Tensor& filter, Tensor* output) {
  std::lock_guard<std::mutex> guard(lock_);

  float* resconv[dnnResourceNumber] = {0};

  dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDst], layout_forward_dst_);
  dnnAllocateBuffer_F32((void**)&resconv[dnnResourceSrc], layout_forward_src_);

  dnnAllocateBuffer_F32((void**)&resconv[dnnResourceFilter], layout_forward_filter_);

  src_convert_MKL(const_cast<Tensor&> (input), resconv[dnnResourceSrc], params_, false, params_.data_format, layout_forward_src_);

  filters_convert_MKL(const_cast<Tensor&>(filter), resconv[dnnResourceFilter],
      params_, false, layout_forward_filter_);
  dnnExecute_F32(forward_, (void**)resconv);
  dst_convert_MKL(*output, resconv[dnnResourceDst], _outputSize, true,
      params_.data_format, layout_forward_dst_);
  dnnReleaseBuffer_F32((void*)resconv[dnnResourceDst]);
  dnnReleaseBuffer_F32((void*)resconv[dnnResourceSrc]);
  dnnReleaseBuffer_F32((void*)resconv[dnnResourceFilter]);
}

void MklConvolution::executeBackwardsFilter(const Tensor& input, const Tensor& out_backprop, Tensor* filter_backprop) {
  std::lock_guard<std::mutex> guard(lock_);

  float* resconv[dnnResourceNumber] = {0};

  dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDiffFilter], layout_backward_filter_filter_);
  dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDiffDst],layout_backward_filter_dst_);
  dnnAllocateBuffer_F32((void**)&resconv[dnnResourceSrc],layout_backward_filter_src_);
  src_convert_MKL(const_cast<Tensor&>(input), resconv[dnnResourceSrc], params_, false, params_.data_format, layout_backward_filter_src_);

  dst_convert_MKL(const_cast<Tensor&>(out_backprop), resconv[dnnResourceDiffDst], _outputSize, false, params_.data_format, layout_backward_filter_dst_);
  dnnExecute_F32(backward_filter_, (void**)resconv);
  filters_convert_MKL(*filter_backprop, resconv[dnnResourceDiffFilter], params_, true, layout_backward_filter_filter_);
  dnnReleaseBuffer_F32((void*)resconv[dnnResourceDiffFilter]);
  dnnReleaseBuffer_F32((void*)resconv[dnnResourceDiffDst]);
  dnnReleaseBuffer_F32((void*)resconv[dnnResourceSrc]);
}

void MklConvolution::executeBackwardsData(const Tensor& filter, const Tensor& out_backprop, Tensor* input_backprop) {
  std::lock_guard<std::mutex> guard(lock_);

  float* resconv[dnnResourceNumber] = {0};

  dnnAllocateBuffer_F32((void**) &resconv[dnnResourceFilter], layout_backward_data_filter_);
  dnnAllocateBuffer_F32((void**) &resconv[dnnResourceDiffDst], layout_backward_data_dst_);
  dnnAllocateBuffer_F32((void**) &resconv[dnnResourceDiffSrc], layout_backward_data_src_);

  filters_convert_MKL(const_cast<Tensor&>(filter), resconv[dnnResourceFilter], params_, false, layout_backward_data_filter_);
  dst_convert_MKL(const_cast<Tensor&>(out_backprop), resconv[dnnResourceDiffDst],
      _outputSize, false, params_.data_format, layout_backward_data_dst_);
  dnnExecute_F32(backward_data_, (void**)resconv);

  src_convert_MKL(*input_backprop, resconv[dnnResourceDiffSrc], params_, true, params_.data_format,
          layout_backward_data_src_);

  dnnReleaseBuffer_F32((void*) resconv[dnnResourceFilter]);
  dnnReleaseBuffer_F32((void*) resconv[dnnResourceDiffDst]);
  dnnReleaseBuffer_F32((void*) resconv[dnnResourceDiffSrc]);
}

void MklConvolution::computeMklParams() {
  fillInputSize(_inputSize);
  fillConvolutionStride(_convolutionStride);
  fillOutputSize(_outputSize, _inputSize, _convolutionStride, params_.padding);
  fillFilterSize(_filterSize);
  fillOffset(_inputSize, _outputSize, _filterSize, _convolutionStride,
      params_.padding, _offset);
  fillConvolutionStride(_convolutionStride);
}

void MklConvolution::createMklPrimitives() {
  int res = 0;
  res = dnnConvolutionCreateForward_F32(&forward_, NULL,
      dnnAlgorithmConvolutionDirect, dimension, _inputSize,
      _outputSize, _filterSize, _convolutionStride, _offset,
      dnnBorderZeros);

  if (res != 0) {
    LOG(WARNING) << "Could not create forward convolution";
  }

  res = dnnConvolutionCreateBackwardFilter_F32(&backward_filter_, NULL,
      dnnAlgorithmConvolutionDirect, dimension, _inputSize,
      _outputSize, _filterSize, _convolutionStride, _offset,
      dnnBorderZeros);

  if (res != 0) {
    LOG(WARNING) << "Could not create backward filter convolution";
  }

  res &= dnnConvolutionCreateBackwardData_F32(&backward_data_, NULL,
      dnnAlgorithmConvolutionDirect, dimension, _inputSize,
      _outputSize, _filterSize, _convolutionStride, _offset,
      dnnBorderZeros);

  if (res != 0) {
    LOG(WARNING) << "Could not create backward data convolution";
  }
}

void MklConvolution::createLayouts() {
  dnnLayoutCreateFromPrimitive_F32(&layout_forward_dst_, forward_, dnnResourceDst);
  dnnLayoutCreateFromPrimitive_F32(&layout_forward_src_, forward_, dnnResourceSrc);
  dnnLayoutCreateFromPrimitive_F32(&layout_forward_filter_, forward_, dnnResourceFilter);

  dnnLayoutCreateFromPrimitive_F32(&layout_backward_filter_filter_, backward_filter_, dnnResourceDiffFilter);
  dnnLayoutCreateFromPrimitive_F32(&layout_backward_filter_dst_, backward_filter_, dnnResourceDiffDst);
  dnnLayoutCreateFromPrimitive_F32(&layout_backward_filter_src_, backward_filter_, dnnResourceSrc);

  dnnLayoutCreateFromPrimitive_F32(&layout_backward_data_filter_, backward_data_, dnnResourceFilter);
  dnnLayoutCreateFromPrimitive_F32(&layout_backward_data_dst_, backward_data_, dnnResourceDiffDst);
  dnnLayoutCreateFromPrimitive_F32(&layout_backward_data_src_, backward_data_, dnnResourceDiffSrc);
}

void MklConvolution::fillInputSize(size_t* inputSize) {
  inputSize[0] = params_.h;
  inputSize[1] = params_.w;
  inputSize[2] = params_.ic;
  inputSize[3] = params_.minibatch;
}

void MklConvolution::fillOutputSize(size_t* outputSize, const size_t* inputSize,
  const size_t* convolutionStride, bool padding) { // TODO use the "padding" from the object

  if (padding) {
    //SAME padding
    outputSize[0] = ceil((double)inputSize[0] / (double)convolutionStride[0]);
    outputSize[1] = ceil((double)inputSize[1] / (double)convolutionStride[1]);
  } else {
    //VALID padding
    outputSize[0] = ceil(double(inputSize[0] - params_.kh + 1) / double(convolutionStride[0]));
    outputSize[1] = ceil(double(inputSize[1] - params_.kw + 1) / double(convolutionStride[1]));
  }

  outputSize[2] = params_.oc;
  outputSize[3] = params_.minibatch;
}

void MklConvolution::fillFilterSize(size_t* filterSize) {
  filterSize[0] = params_.kh;
  filterSize[1] = params_.kw;
  filterSize[2] = params_.ic;
  filterSize[3] = params_.oc;
}

void MklConvolution::fillOffset(const size_t* inputSize, const size_t* outputSize,
  const size_t* filterSize, const size_t* convolutionStride,
  bool padding, int* offset) {

  const int dimensions = 2;
  for (int i = 0; i != dimensions; ++i) {
    if (padding) {
      int missing_pixels = outputSize[i] * convolutionStride[i]
                                   + (filterSize[i] - convolutionStride[i])
                                   - inputSize[i];
      offset[i] =  - missing_pixels / 2;
    } else {
      offset[i] = 0;
    }
  }
}

void MklConvolution::fillConvolutionStride(size_t* convolutionStride) {
  convolutionStride[0] = params_.c_stride;
  convolutionStride[1] = params_.c_stride;
}
} //namespace tensorflow
