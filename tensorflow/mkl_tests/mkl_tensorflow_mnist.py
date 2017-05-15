import argparse
from tensorflow.python.client import timeline
import time
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mkl', help='use the Intel MKL 2D convolution', default=False, type=int)
parser.add_argument('--batch', help='batch size', default=1024, type=int)
parser.add_argument('--ic', help='input channels', default=16, type=int)
parser.add_argument('--tf_log', help='tf logging level', default=1, type=int)
parser.add_argument('--max_epoch', help='number of epochs', default=20000, type=int)


args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.framework import ops

if args.mkl:
    print "using MKL convolution"
    label_map = {"Conv2D": "MKL",
                 "Conv2DBackpropFilter": "MKL",
                 "Conv2DBackpropInput": "MKL"}
else:
    label_map = {}


with ops.Graph().as_default() as g:
    with g._kernel_label_map(label_map):
        
        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)
        
        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)
          
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        
        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        
        input_channels = args.ic
        # INPUT DATA
        x = tf.placeholder(tf.float32, shape=[None, 784 * input_channels])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        # FIRST CONVOLUTIONAL LAYER
        
        ks = 5
        
        W_conv1 = weight_variable([ks, ks, input_channels, 32])
        b_conv1 = bias_variable([32])
        

        x_image = tf.reshape(x, [-1,28,28,input_channels])
        
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        
        # SECOND CONVOLUTIONAL LAYER
        
        W_conv2 = weight_variable([ks, ks, 32, 64])
        b_conv2 = bias_variable([64])
        
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        
        # FULLY CONNECTED LAYER
        
        #
        W_fc1 = weight_variable([4 * 4 * 64, 1024])
        b_fc1 = bias_variable([1024])
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # DROPOUT>
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        # SOFTMAX
        
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        
        
        learning_rate = 1e-4
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        
        print_freq = 1

        t1 = 0
        t2 = 0        
        batch_size = args.batch
        run_options = None
        run_metadata = None
        with sess.as_default():
            for i in range(args.max_epoch):
              batch = mnist.train.next_batch(batch_size)
              x_dummy = np.zeros((batch_size, 28*28*(input_channels - 1)))
              x_ = np.concatenate([batch[0], x_dummy], axis=1)
              if i%print_freq == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:x_, y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g, time: %g"%(i, train_accuracy, t2 - t1))
              t1 = time.time()
              
              sess.run([train_step], feed_dict={x: x_, y_: batch[1], keep_prob: 0.5},
                             options=run_options, run_metadata=run_metadata)
              t2 = time.time()
               
            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
