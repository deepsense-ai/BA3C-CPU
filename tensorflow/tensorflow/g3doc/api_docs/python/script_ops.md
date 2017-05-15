<!-- This file is machine generated: DO NOT EDIT! -->

# Wraps python functions

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Script Language Operators.

TensorFlow provides allows you to wrap python/numpy functions as
TensorFlow operators.

- - -

### `tf.py_func(func, inp, Tout, stateful=True, name=None)` {#py_func}

Wraps a python function and uses it as a tensorflow op.

Given a python function `func`, which takes numpy arrays as its
inputs and returns numpy arrays as its outputs. E.g.,

```python
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.sinh(x)
inp = tf.placeholder(tf.float32, [...])
y = py_func(my_func, [inp], [tf.float32])
```

The above snippet constructs a tf graph which invokes a numpy
sinh(x) as an op in the graph.

##### Args:


*  <b>`func`</b>: A python function.
*  <b>`inp`</b>: A list of `Tensor`.
*  <b>`Tout`</b>: A list or tuple of tensorflow data types or a single tensorflow data
        type if there is only one, indicating what `func` returns.
*  <b>`stateful`</b>: A boolean indicating whether the function should be considered
            stateful or stateless. I.e. whether it, given the same input, will
            return the same output and at the same time does not change state
            in an observable way. Optimizations such as common subexpression
            elimination are only possible when operations are stateless.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list of `Tensor` or a single `Tensor` which `func` computes.


