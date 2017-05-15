"""
This file contains tests that check if the convolutions provided
by the MKL-enabled tensorflow are numerically equivalent to the
default tensorflow implementation

@author Tomasz Grel, tomasz.grel@codilime.com
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import unittest
import time 

def _ones_images(width, height, batches, channels):
    input_data = np.ones(shape=(batches, width, height, channels))
    return input_data

def _ones_filters(f_width, f_height, in_channels, out_channels):
    filters = np.ones(shape=(f_width, f_height, in_channels, out_channels))
    return filters

def _random_images(width, height, batches, channels):
    input_data = 10 * np.random.uniform(size=(batches, width, height, channels))
    return input_data

def _random_filters(f_width, f_height, in_channels, out_channels):
    filters = np.random.uniform(size=(f_width, f_height, in_channels, out_channels))
    return filters

def _convolution(input_data, filters, params):
    data = tf.placeholder("float")
    conv_filter = tf.placeholder("float")
    c = tf.nn.conv2d(data, conv_filter, **params)

    conv_filter_grad = tf.gradients(c, [conv_filter])[0]
    conv_input_grad = tf.gradients(c, [data])[0]
    init = tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=1,
                                use_per_session_threads=True)) as sess:
        sess.run(init)                
        feed_dict={data: input_data, conv_filter: filters}
        [c_out, df_out, dd_out] = sess.run([c, conv_filter_grad, conv_input_grad], feed_dict=feed_dict)
    return c_out, df_out, dd_out


class MklConvolutionTestBase(object):
    def test_2x2_filter(self):
        in_channels = 1
        input_data = _random_images(width=100, height=100, batches=1, channels=in_channels) 
        filters = _random_filters(f_width=2, f_height=2, in_channels=in_channels, out_channels=4)
        self.check_against_tensorflow(input_data, filters)

    
    def test_one_batch_one_channel(self):
        in_channels = 1
        input_data = _random_images(width=100, height=100, batches=1, channels=in_channels) 
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=4)
        self.check_against_tensorflow(input_data, filters)

    def test_one_batch_many_channels(self):
        in_channels = 10
        input_data = _random_images(width=100, height=100, batches=1, channels=in_channels) 
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=4)
        self.check_against_tensorflow(input_data, filters)

    def test_one_channel_many_batches(self):
        in_channels = 1
        input_data = _random_images(width=100, height=100, batches=10, channels=in_channels) 
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=4)
        self.check_against_tensorflow(input_data, filters)

    def test_full(self):
        in_channels = 20
        input_data = _random_images(width=100, height=100, batches=10, channels=in_channels) 
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=10)
        self.check_against_tensorflow(input_data, filters)

    def test_valid_padding(self):
        in_channels = 1
        input_data = _random_images(width=10, height=10, batches=1, channels=in_channels) 
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=1)
        self.check_against_tensorflow(input_data, filters, padding='VALID')
        
    def test_2x2_stride(self):
        in_channels = 1
        input_data = _random_images(width=10, height=10, batches=1, channels=in_channels) 
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=1)
        self.check_against_tensorflow(input_data, filters, stride=[2,2])

    def test_2x2_stride_valid_padding(self):
        in_channels = 1
        input_data = _random_images(width=10, height=10, batches=1, channels=in_channels) 
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=1)
        self.check_against_tensorflow(input_data, filters, padding='VALID', stride=[2,2])

    def test_nonsquare_input_image(self):
        in_channels = 1
        input_data = _random_images(width=31, height=9, batches=1, channels=in_channels)
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=1)
        self.check_against_tensorflow(input_data, filters)

    def test_nonsquare_input_image2(self):
        in_channels = 1
        input_data = _random_images(width=5, height=30, batches=1, channels=in_channels)
        filters = _random_filters(f_width=3, f_height=3, in_channels=in_channels, out_channels=1)
        self.check_against_tensorflow(input_data, filters)

    def test_16_to_16_channels(self):
        in_channels = 16
        input_data = _random_images(width=4, height=4, batches=1, channels=in_channels)
        filters = _random_filters(f_width=2, f_height=2, in_channels=in_channels, out_channels=16)
        filters = np.arange(filters.size).reshape(filters.shape)
        self.check_against_tensorflow(input_data, filters, padding='VALID')

    def test_16_to_32_channels(self):
        in_channels = 16
        input_data = _random_images(width=4, height=4, batches=1, channels=in_channels)
        filters = _random_filters(f_width=2, f_height=2, in_channels=in_channels, out_channels=16)
        filters = np.arange(filters.size).reshape(filters.shape)
        self.check_against_tensorflow(input_data, filters, padding='VALID')

    def test_15_to_31_channels(self):
        in_channels = 15
        input_data = _random_images(width=4, height=4, batches=1, channels=in_channels)
        filters = _random_filters(f_width=2, f_height=2, in_channels=in_channels, out_channels=31)
        filters = np.arange(filters.size).reshape(filters.shape)
        self.check_against_tensorflow(input_data, filters, padding='VALID')


    def test_1_21_21_32(self):
        in_channels = 32
        input_data = _random_images(width=21, height=21, batches=1, channels=in_channels)
        filters = _random_filters(f_width=4, f_height=4, in_channels=in_channels, out_channels=64)
        self.check_against_tensorflow(input_data, filters, padding='VALID')
        
        
    def test_mnist(self):
        in_channels = 1
        input_data = _random_images(width=28, height=28, batches=50, channels=in_channels)
        filters = _random_filters(f_width=6, f_height=6, in_channels=in_channels, out_channels=32)
        self.check_against_tensorflow(input_data, filters, padding='VALID')
        

class MklConvolutionTestForward(MklConvolutionTestBase, unittest.TestCase):
    def check_against_tensorflow(self, input_data, filters, padding='VALID', stride=[1,1],
                                 data_format="NHWC"):
        params = {"strides": [1, stride[0], stride[1], 1],
                  "padding": padding,
                  "data_format": data_format}
        with ops.Graph().as_default() as g:
            with g._kernel_label_map({"Conv2D" : "MKL"}):
                c_mkl, _, _ = _convolution(input_data, filters, params)
            with g._kernel_label_map({}):            
                c_tf, _, _ = _convolution(input_data, filters, params)

            errors = np.abs(c_tf - c_mkl)
            
            rel_errors = np.abs(c_tf - c_mkl) / np.abs(c_tf)
            self.assertTrue(rel_errors.max() < 1e-5)

class MklConvolutionTestBackwardData(MklConvolutionTestBase, unittest.TestCase):
    def check_against_tensorflow(self, input_data, filters, padding='SAME', stride=[1, 1], data_format="NHWC"):
        params = {"strides": [1, stride[0], stride[1], 1],
                  "padding": padding,
                  "data_format": data_format}
        
        with ops.Graph().as_default() as g:
            with g._kernel_label_map({"Conv2DBackpropInput" : "MKL"}):
                _, _, dd_mkl = _convolution(input_data, filters, params)
            with g._kernel_label_map({}):            
                _, _, dd_tf = _convolution(input_data, filters, params)

            errors = np.abs(dd_tf - dd_mkl)
            
            rel_errors = np.abs(dd_tf - dd_mkl) / np.abs(dd_tf)
            if not np.isnan(rel_errors.max()):
                self.assertLess(rel_errors.max(), 10e-5)
            else: 
                self.assertLess(errors.max(), 10e-5)
                    

class MklConvolutionTestBackwardFilter(MklConvolutionTestBase, unittest.TestCase):
    def check_against_tensorflow(self, input_data, filters, padding='SAME', stride=[1,1], data_format="NHWC"):
        params = {"strides": [1, stride[0], stride[1], 1],
                  "padding": padding,
                  "data_format": data_format}
        
        with ops.Graph().as_default() as g:
            with g._kernel_label_map({"Conv2DBackpropFilter" : "MKL"}):
                _, df_mkl, _ = _convolution(input_data, filters, params)
            with g._kernel_label_map({}):            
                _, df_tf, _ = _convolution(input_data, filters, params)

            errors = np.abs(df_tf - df_mkl)
            
            rel_errors = np.abs(df_tf - df_mkl) / np.abs(df_tf)
            
            if not np.isnan(rel_errors.max()):
                self.assertLess(rel_errors.max(), 10e-5)
            else: 
                self.assertLess(errors.max(), 10e-5)

if __name__ == '__main__':
    unittest.main()
        