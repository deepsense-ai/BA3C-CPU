# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for py_func op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import queue
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import script_ops


class PyOpTest(tf.test.TestCase):

  def testBasic(self):

    def my_func(x, y):
      return np.sinh(x) + np.cosh(y)

    # single type
    with self.test_session():
      x = tf.constant(1.0, tf.float32)
      y = tf.constant(2.0, tf.float32)
      z = tf.py_func(my_func, [x, y], tf.float32)
      self.assertEqual(z.eval(), my_func(1.0, 2.0).astype(np.float32))

    # scalar
    with self.test_session():
      x = tf.constant(1.0, tf.float32)
      y = tf.constant(2.0, tf.float32)
      z = tf.py_func(my_func, [x, y], [tf.float32])
      self.assertEqual(z[0].eval(), my_func(1.0, 2.0).astype(np.float32))

    # array
    with self.test_session():
      x = tf.constant([1.0, 2.0], tf.float64)
      y = tf.constant([2.0, 3.0], tf.float64)
      z = tf.py_func(my_func, [x, y], [tf.float64])
      self.assertAllEqual(
          z[0].eval(),
          my_func([1.0, 2.0], [2.0, 3.0]).astype(np.float64))

    # a bit exotic type (complex64)
    with self.test_session():
      x = tf.constant(1+2j, tf.complex64)
      y = tf.constant(3+4j, tf.complex64)
      z, = tf.py_func(my_func, [x, y], [tf.complex64])
      self.assertAllClose(z.eval(), my_func(1+2j, 3+4j))

    # a bit excotic function (rfft)
    with self.test_session():
      x = tf.constant([1., 2., 3., 4.], tf.float32)
      def rfft(x):
        return np.fft.rfft(x).astype(np.complex64)
      y, = tf.py_func(rfft, [x], [tf.complex64])
      self.assertAllClose(y.eval(), np.fft.rfft([1., 2., 3., 4.]))

    # returns a python literal.
    with self.test_session():
      def literal(x):
        return 1.0 if x == 0.0 else 0.0
      x = tf.constant(0.0, tf.float64)
      y, = tf.py_func(literal, [x], [tf.float64])
      self.assertAllClose(y.eval(), 1.0)

    # returns a list
    with self.test_session():
      def list_func(x):
        return [x, x + 1]
      x = tf.constant(0.0, tf.float64)
      y, z = tf.py_func(list_func, [x], [tf.float64] * 2)
      self.assertAllClose(y.eval(), 0.0)
      self.assertAllClose(z.eval(), 1.0)

    # returns a tuple
    with self.test_session():
      def tuple_func(x):
        return x, x + 1
      x = tf.constant(0.0, tf.float64)
      y, z = tf.py_func(tuple_func, [x], [tf.float64] * 2)
      self.assertAllClose(y.eval(), 0.0)
      self.assertAllClose(z.eval(), 1.0)

    # returns a tuple, Tout and inp a tuple
    with self.test_session():
      x = tf.constant(0.0, tf.float64)
      y, z = tf.py_func(tuple_func, (x,), (tf.float64, tf.float64))
      self.assertAllClose(y.eval(), 0.0)
      self.assertAllClose(z.eval(), 1.0)

  def testStrings(self):

    def read_fixed_length_numpy_strings():
      return np.array([b" there"])

    def read_and_return_strings(x, y):
      return x + y

    with self.test_session():
      x = tf.constant([b"hello", b"hi"], tf.string)
      y, = tf.py_func(read_fixed_length_numpy_strings, [], [tf.string])
      z, = tf.py_func(read_and_return_strings, [x, y], [tf.string])
      self.assertListEqual(list(z.eval()), [b"hello there", b"hi there"])

  def testStringPadding(self):
    correct = [b"this", b"is", b"a", b"test"]
    with self.test_session():
      s, = tf.py_func(lambda: [correct], [], [tf.string])
      self.assertAllEqual(s.eval(), correct)

  def testLarge(self):
    with self.test_session() as sess:
      x = tf.zeros([1000000], dtype=np.float32)
      y = tf.py_func(lambda x: x + 1, [x], [tf.float32])
      z = tf.py_func(lambda x: x * 2, [x], [tf.float32])
      for _ in xrange(100):
        sess.run([y[0].op, z[0].op])

  def testNoInput(self):
    with self.test_session():
      x, = tf.py_func(lambda: 42.0, [], [tf.float64])
      self.assertAllClose(x.eval(), 42.0)

  def testCleanup(self):
    for _ in xrange(1000):
      g = tf.Graph()
      with g.as_default():
        c = tf.constant([1.], tf.float32)
        _ = tf.py_func(lambda x: x + 1, [c], [tf.float32])
    self.assertTrue(script_ops._py_funcs.size() < 100)

  def testBadNumpyReturnType(self):
    with self.test_session():

      def bad():
        # Structured numpy arrays aren't supported.
        return np.array([], dtype=[("foo", np.float32)])

      y, = tf.py_func(bad, [], [tf.float32])

      with self.assertRaisesRegexp(errors.UnimplementedError,
                                   "Unsupported numpy type"):
        y.eval()

  def testBadReturnType(self):
    with self.test_session():

      def bad():
        # Non-string python objects aren't supported.
        return tf.float32

      z, = tf.py_func(bad, [], [tf.float64])

      with self.assertRaisesRegexp(errors.UnimplementedError,
                                   "Unsupported object type"):
        z.eval()

  def testStateful(self):
    # Not using self.test_session(), which disables optimization.
    with tf.Session() as sess:
      producer = iter(range(3))
      x, = tf.py_func(lambda: next(producer), [], [tf.int64])
      self.assertEqual(sess.run(x), 0)
      self.assertEqual(sess.run(x), 1)
      self.assertEqual(sess.run(x), 2)

  def testStateless(self):
    # Not using self.test_session(), which disables optimization.
    with tf.Session() as sess:
      producer = iter(range(3))
      x, = tf.py_func(lambda: next(producer), [], [tf.int64], stateful=False)
      self.assertEqual(sess.run(x), 0)
      self.assertEqual(sess.run(x), 0)
      self.assertEqual(sess.run(x), 0)

  def testGradientFunction(self):
    # Input to tf.py_func is necessary, otherwise get_gradient_function()
    # returns None per default.
    a = tf.constant(0)
    x, = tf.py_func(lambda a: 0, [a], [tf.int64])
    y, = tf.py_func(lambda a: 0, [a], [tf.int64], stateful=False)
    self.assertEqual(None, ops.get_gradient_function(x.op))
    self.assertEqual(None, ops.get_gradient_function(y.op))

  def testCOrder(self):
    with self.test_session():
      val = [[1, 2], [3, 4]]
      x, = tf.py_func(lambda: np.array(val, order="F"), [], [tf.int64])
      self.assertAllEqual(val, x.eval())

  def testParallel(self):
    # Tests that tf.py_func's can run in parallel if they release the GIL.
    with self.test_session() as session:
      q = queue.Queue(1)

      def blocking_put():
        q.put(42)
        q.join()  # Wait for task_done().
        return 42

      def blocking_get():
        v = q.get(block=True)  # Wait for put().
        q.task_done()
        return v

      x, = tf.py_func(blocking_put, [], [tf.int64])
      y, = tf.py_func(blocking_get, [], [tf.int64])

      # This will result in a deadlock if the py_func's don't run in parallel.
      session.run([x, y])

  def testNoReturnValueStateful(self):

    class State(object):

      def __init__(self):
        self._value = np.array([1], np.int64)

      def _increment(self, diff):
        self._value += diff

      def increment(self, diff):
        return tf.py_func(self._increment, [diff], [], stateful=True)

      @property
      def value(self):
        return self._value

    with self.test_session() as sess:
      s = State()
      op = s.increment(tf.constant(2, tf.int64))
      ret = sess.run(op)
      self.assertIsNone(ret)
      self.assertAllEqual([3], s.value)

  def testNoReturnValueStateless(self):

    def do_nothing(unused_x):
      pass

    f = tf.py_func(do_nothing, [tf.constant(3, tf.int64)], [], stateful=False)
    with self.test_session() as sess:
      self.assertEqual(sess.run(f), [])


if __name__ == "__main__":
  tf.test.main()
