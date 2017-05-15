# -*- coding: UTF-8 -*-
# File: symbolic_functions.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
from ..utils import logger

def prediction_incorrect(logits, label, topk=1):
    """
    :param logits: NxC
    :param label: N
    :returns: a float32 vector of length N with 0/1 values, 1 meaning incorrect prediction
    """
    return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)), tf.float32)

def flatten(x):
    """
    Flatten the tensor.
    """
    return tf.reshape(x, [-1])

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, np.prod(shape)])
    return tf.reshape(x, tf.pack([tf.shape(x)[0], -1]))

def logSoftmax(x):
    """
    Batch log softmax.
    :param x: NxC tensor.
    :returns: NxC tensor.
    """
    logger.warn("symbf.logSoftmax is deprecated in favor of tf.nn.log_softmax")
    return tf.nn.log_softmax(x)

def class_balanced_binary_class_cross_entropy(pred, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss for binary classification,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    :param pred: size: b x ANYTHING. the predictions in [0,1].
    :param label: size: b x ANYTHING. the ground truth in {0,1}.
    :returns: class-balanced binary classification cross entropy loss
    """
    z = batch_flatten(pred)
    y = tf.cast(batch_flatten(label), tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    eps = 1e-8
    loss_pos = -beta * tf.reduce_mean(y * tf.log(tf.abs(z) + eps), 1)
    loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(tf.abs(1. - z) + eps), 1)
    cost = tf.sub(loss_pos, loss_neg)
    cost = tf.reduce_mean(cost, name=name)
    return cost

def print_stat(x, message=None):
    """ a simple print op.
        Use it like: x = print_stat(x)
    """
    if message is None:
        message = x.op.name
    return tf.Print(x, [tf.reduce_mean(x), x], summarize=20,
            message=message, name='print_' + x.op.name)

def rms(x, name=None):
    if name is None:
        name = x.op.name + '/rms'
        with tf.name_scope(None):   # name already contains the scope
            return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)
    return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)

def huber_loss(x, delta=1, name=None):
    if name is None:
        name = 'huber_loss'
    sqrcost = tf.square(x)
    abscost = tf.abs(x)
    return tf.reduce_sum(
            tf.select(abscost < delta,
                sqrcost * 0.5,
                abscost * delta - 0.5 * delta ** 2),
            name=name)

def get_scalar_var(name, init_value):
    return tf.get_variable(name, shape=[],
            initializer=tf.constant_initializer(init_value),
            trainable=False)
