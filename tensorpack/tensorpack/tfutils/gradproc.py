#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gradproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import ABCMeta, abstractmethod
import re
import inspect
from ..utils import logger
from .symbolic_functions import rms
from .summary import add_moving_summary

__all__ = ['GradientProcessor', 'SummaryGradient', 'CheckGradient',
           'ScaleGradient', 'MapGradient']

class GradientProcessor(object):
    __metaclass__ = ABCMeta

    def process(self, grads):
        """
        Process the symbolic gradients.

        :param grads: list of (grad, var)
        :returns: symbolic gradients with the same type as input
        """
        with tf.name_scope(type(self).__name__):
            return self._process(grads)

    @abstractmethod
    def _process(self, grads):
        pass

class MapGradient(GradientProcessor):
    """
    Apply a function on all gradient if the name matches regex.
    Keep the other gradients unchanged.
    """
    def __init__(self, func, regex='.*'):
        """
        :param func: takes a grad or (grad, var) pair and returns a grad. If return None, the
            gradient is discarded.
        :param regex: used to match variables. default to match all variables.
        """
        args = inspect.getargspec(func).args
        arg_num = len(args) - inspect.ismethod(func)
        assert arg_num in [1, 2], \
                "The function must take 1 or 2 arguments!  ({})".format(args)
        if arg_num == 1:
            self.func = lambda grad, var: func(grad)
        else:
            self.func = func

        if not regex.endswith('$'):
            regex = regex + '$'
        self.regex = regex

    def _process(self, grads):
        ret = []
        for grad, var in grads:
            if re.match(self.regex, var.op.name):
                grad = self.func(grad, var)
                if grad is not None:
                    ret.append((grad, var))
            else:
                ret.append((grad, var))
        return ret

_summaried_gradient = set()

class SummaryGradient(MapGradient):
    """
    Summary history and RMS for each graident variable
    """
    def __init__(self):
        super(SummaryGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        name = var.op.name
        if name not in _summaried_gradient:
            _summaried_gradient.add(name)
            tf.histogram_summary(name + '/grad', grad)
            add_moving_summary(rms(grad, name=name + '/rms'))
        return grad

class CheckGradient(MapGradient):
    """
    Check for numeric issue.
    """
    def __init__(self):
        super(CheckGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        # this is very slow...
        #op = tf.Assert(tf.reduce_all(tf.is_finite(var)), [var], summarize=100)
        grad = tf.check_numerics(grad, 'CheckGradient')
        return grad

class ScaleGradient(MapGradient):
    """
    Scale gradient by a multiplier
    """
    def __init__(self, multipliers):
        """
        :param multipliers: list of (regex, float)
        """
        self.multipliers = multipliers
        super(ScaleGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        varname = var.op.name
        for regex, val in self.multipliers:
            # always match against the whole name
            if not regex.endswith('$'):
                regex = regex + '$'

            if re.match(regex, varname):
                logger.info("Apply lr multiplier {} for {}".format(val, varname))
                if val != 0:    # skip zero to speed up
                    return grad * val
                else:
                    return None
        return grad
