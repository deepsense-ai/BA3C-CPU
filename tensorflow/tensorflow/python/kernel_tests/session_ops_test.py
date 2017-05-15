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
"""Tests for tensorflow.ops.session_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SessionOpsTest(tf.test.TestCase):

  def testHandleBasic(self):
    with self.test_session() as sess:
      # Return a handle.
      a = tf.constant(10)
      b = tf.constant(5)
      c = tf.mul(a, b)
      h = tf.get_session_handle(c)
      h = sess.run(h)

      # Feed a tensor handle.
      f, x = tf.get_session_tensor(h.handle, tf.int32)
      y = tf.mul(x, 10)
      self.assertEqual(500, sess.run(y, feed_dict={f: h.handle}))

  def testHandleEval(self):
    with self.test_session() as sess:
      # Return a handle.
      a = tf.constant(10)
      b = tf.constant(5)
      c = tf.mul(a, b)
      h = tf.get_session_handle(c)
      h = sess.run(h)

      # Get the tensor from its handle.
      self.assertEqual(50, h.eval())

  def testHandleAndValue(self):
    with self.test_session() as sess:
      # Return a handle and a value.
      a = tf.constant(10)
      b = tf.constant(5)
      c = tf.mul(a, b)
      h = tf.get_session_handle(c)
      v = tf.mul(a, c)
      h, v = sess.run([h, v])

      self.assertEqual(50, h.eval())
      self.assertEqual(500, v)

  def testHandleCond(self):
    with self.test_session() as sess:
      # Return a handle and a value
      a = tf.constant(10)
      b = tf.constant(5)
      p = tf.less(a, b)
      c = tf.mul(a, b)
      h = tf.get_session_handle(c)
      p, h = sess.run([p, h])

      # Run by feeding a tensor handle.
      f, x = tf.get_session_tensor(h.handle, tf.int32)
      if p:
        y = tf.mul(x, 10)
      else:
        y = tf.mul(x, 100)
      result = sess.run(y, feed_dict={f: h.handle})

      self.assertEqual(5000, result)

  def testHandleForLoop(self):
    with self.test_session() as sess:
      # Initialize a handle.
      a = tf.constant(0)
      h = tf.get_session_handle(a)
      h = sess.run(h)

      # Do some computation.
      f, x = tf.get_session_tensor(h.handle, tf.int32)
      # Must define the loop body outside the loop.
      h_x = tf.get_session_handle(tf.add(x, 1))
      for _ in range(100):
        # This exercises garbage collection.
        h = sess.run(h_x, feed_dict={f: h.handle})

      self.assertEqual(100, h.eval())

  def testHandleWhileLoop(self):
    with self.test_session() as sess:
      # Initialize a handle.
      a = tf.constant(0)
      h = tf.get_session_handle(a)
      h = sess.run(h)

      # Do some computation.
      f, x = tf.get_session_tensor(h.handle, tf.int32)
      b = tf.constant(100)
      p = tf.less(x, b)
      # Must define the loop body outside the loop.
      h_x = tf.get_session_handle(tf.add(x, 1))
      while True:
        rp, h = sess.run([p, h_x], feed_dict={f: h.handle})
        if not rp:
          break

      self.assertEqual(101, h.eval())

  def testHandleMover(self):
    with self.test_session() as sess:
      # Return a handle.
      a = tf.constant(10)
      b = tf.constant(5)
      c = tf.mul(a, b)
      h = tf.get_session_handle(c)
      h = sess.run(h)

      # Feed a tensor handle.
      f, x = tf.get_session_tensor(h.handle, tf.int32)
      y = tf.mul(x, 10)
      self.assertEqual(500, sess.run(y, feed_dict={f: h.handle}))

      # Feed another tensor handle.
      with tf.device("/gpu:0"):
        a = tf.constant(10)
        h = tf.get_session_handle(a)
        h = sess.run(h)
        self.assertEqual(100, sess.run(y, feed_dict={f: h.handle}))

  def testHandleDelete(self):
    with self.test_session() as sess:
      # Return a handle.
      a = tf.constant(10)
      b = tf.constant(5)
      c = tf.mul(a, b)
      h = tf.get_session_handle(c)
      sess.run(h).delete()

  def testHandleDeleteRaw(self):
    with self.test_session() as sess:
      # Return a handle.
      a = tf.constant(10)
      b = tf.constant(5)
      c = tf.mul(a, b)
      h = tf.get_session_handle(c)
      h = sess.run(h)

      # Delete using a raw tensor handle.
      raw_h = h.get_raw_handle()
      f, x = tf.delete_session_tensor(raw_h)
      sess.run(x, feed_dict={f: raw_h})

  def testMultiDevices(self):
    with self.test_session() as sess:
      with tf.device("/gpu:0"):
        a = tf.constant(1.0)
        a_handle = sess.run(tf.get_session_handle(a))
      with tf.device("/cpu:0"):
        b = tf.constant(2.0)
        b_handle = sess.run(tf.get_session_handle(b))

      a_p, a_t = tf.get_session_tensor(a_handle.handle, tf.float32)
      b_p, b_t = tf.get_session_tensor(b_handle.handle, tf.float32)
      c = tf.add(a_t, b_t)
      c_handle = sess.run(
          tf.get_session_handle(c),
          feed_dict={a_p: a_handle.handle,
                     b_p: b_handle.handle})
      self.assertEqual(3.0, c_handle.eval())

  def testHandleGC(self):
    with self.test_session() as sess:
      # initial values live on CPU
      with tf.device("/cpu:0"):
        one = tf.constant(1, dtype=tf.float32)
        one_handle = sess.run(tf.get_session_handle(one))
        x_handle = sess.run(tf.get_session_handle(one))

      # addition lives on GPU
      with tf.device("/gpu:0"):
        add_h1, add_t1 = tf.get_session_tensor(one_handle.handle, tf.float32)
        add_h2, add_t2 = tf.get_session_tensor(x_handle.handle, tf.float32)
        add_op = tf.add(add_t1, add_t2)
        add_output = tf.get_session_handle(add_op)

      # add 1 to tensor 20 times
      for _ in range(20):
        x_handle = sess.run(add_output,
                            feed_dict={add_h1: one_handle.handle,
                                       add_h2: x_handle.handle})

  def testHandlePlacement(self):
    with self.test_session() as sess:
      a = tf.constant(1.0)
      a_handle_op = tf.get_session_handle(a)
      b = tf.constant(2.0)
      b_handle_op = tf.get_session_handle(b)

      a_handle = sess.run(a_handle_op)
      b_handle = sess.run(b_handle_op)

      a_p, a_t = tf.get_session_tensor(a_handle.handle, tf.float32)
      b_p, b_t = tf.get_session_tensor(b_handle.handle, tf.float32)

      c = tf.add(a_t, b_t)
      c_handle = sess.run(
          tf.get_session_handle(c),
          feed_dict={a_p: a_handle.handle,
                     b_p: b_handle.handle})
      self.assertEqual(3.0, c_handle.eval())

if __name__ == "__main__":
  tf.test.main()
