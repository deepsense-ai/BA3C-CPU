# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A collection that allows repeated access to its top-scoring items."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python.ops import topn_ops


class TopN(object):
  """A collection that allows repeated access to its top-scoring items.

  A TopN supports the following three operations:
  1) insert(ids, scores).  ids is a 1-d int64 Tensor and scores is a 1-d
  float Tensor.  scores[i] is the score associated with ids[i].  It is
  totally fine to re-insert ids that have already been inserted into the
  collection.

  2) remove(ids)

  3) ids, scores = get_best(n).  scores will contain the n highest (most
  positive) scores currently in the TopN, and ids their corresponding ids.
  n is a 1-d int32 Tensor with shape (1).

  TopN is implemented using a short-list of the top scoring items.  At
  construction time, the size of the short-list must be specified, and it
  is an error to call GetBest(n) with an n greater than that size.
  """

  def __init__(self, max_id, shortlist_size=100, name_prefix=''):
    """Creates a new TopN."""
    self.ops = topn_ops.Load()
    self.shortlist_size = shortlist_size
    # id_to_score contains all the scores we are tracking.
    self.id_to_score = tf.get_variable(
        name=name_prefix + 'id_to_score',
        dtype=tf.float32,
        shape=[max_id],
        initializer=tf.constant_initializer(tf.float32.min))
    # sl_ids and sl_scores together satisfy four invariants:
    # 1) If sl_ids[i] != -1, then
    #    id_to_score[sl_ids[i]] = sl_scores[i] >= sl_scores[0]
    # 2) sl_ids[0] is the number of i > 0 for which sl_ids[i] != -1.
    # 3) If id_to_score[i] > sl_scores[0], then
    #    sl_ids[j] = i for some j.
    # 4) If sl_ids[i] == -1, then sl_scores[i] = tf.float32.min.
    self.sl_ids = tf.get_variable(
        name=name_prefix + 'shortlist_ids',
        dtype=tf.int64,
        shape=[shortlist_size + 1],
        initializer=tf.constant_initializer(-1))
    # Ideally, we would set self.sl_ids[0] = 0 here.  But then it is hard
    # to pass that control dependency to the other other Ops.  Instead, we
    # have insert, remove and get_best all deal with the fact that
    # self.sl_ids[0] == -1 actually means the shortlist size is 0.
    self.sl_scores = tf.get_variable(
        name=name_prefix + 'shortlist_scores',
        dtype=tf.float32,
        shape=[shortlist_size + 1],
        initializer=tf.constant_initializer(tf.float32.min))
    # TopN keeps track of its internal data dependencies, so the user
    # doesn't have to.
    self.last_ops = []

  def insert(self, ids, scores):
    """Insert the ids and scores into the TopN."""
    with tf.control_dependencies(self.last_ops):
      scatter_op = tf.scatter_update(self.id_to_score, ids, scores)
      larger_scores = tf.greater(scores, self.sl_scores[0])

      def shortlist_insert():
        larger_ids = tf.boolean_mask(tf.to_int64(ids), larger_scores)
        larger_score_values = tf.boolean_mask(scores, larger_scores)
        shortlist_ids, new_ids, new_scores = self.ops.top_n_insert(
            self.sl_ids, self.sl_scores, larger_ids, larger_score_values)
        u1 = tf.scatter_update(self.sl_ids, shortlist_ids, new_ids)
        u2 = tf.scatter_update(self.sl_scores, shortlist_ids, new_scores)
        return tf.group(u1, u2)

      # We only need to insert into the shortlist if there are any
      # scores larger than the threshold.
      cond_op = tf.cond(
          tf.reduce_any(larger_scores), shortlist_insert, tf.no_op)
      with tf.control_dependencies([cond_op]):
        self.last_ops = [scatter_op, cond_op]

  def remove(self, ids):
    """Remove the ids (and their associated scores) from the TopN."""
    with tf.control_dependencies(self.last_ops):
      scatter_op = tf.scatter_update(
          self.id_to_score,
          ids,
          tf.ones_like(
              ids, dtype=tf.float32) * tf.float32.min)
      # We assume that removed ids are almost always in the shortlist,
      # so it makes no sense to hide the Op behind a tf.cond
      shortlist_ids_to_remove, new_length = self.ops.top_n_remove(self.sl_ids,
                                                                  ids)
      u1 = tf.scatter_update(
          self.sl_ids, tf.concat(0, [[0], shortlist_ids_to_remove]),
          tf.concat(0, [new_length,
                        tf.ones_like(shortlist_ids_to_remove) * -1]))
      u2 = tf.scatter_update(
          self.sl_scores,
          shortlist_ids_to_remove,
          tf.float32.min * tf.ones_like(
              shortlist_ids_to_remove, dtype=tf.float32))
      self.last_ops = [scatter_op, u1, u2]

  def get_best(self, n):
    """Return the indices and values of the n highest scores in the TopN."""

    def refresh_shortlist():
      """Update the shortlist with the highest scores in id_to_score."""
      new_scores, new_ids = tf.nn.top_k(self.id_to_score, self.shortlist_size)
      smallest_new_score = tf.reduce_min(new_scores)
      new_length = tf.reduce_sum(
          tf.to_int32(tf.greater(new_scores, tf.float32.min)))
      u1 = self.sl_ids.assign(
          tf.to_int64(tf.concat(0, [[new_length], new_ids])))
      u2 = self.sl_scores.assign(
          tf.concat(0, [[smallest_new_score], new_scores]))
      self.last_ops = [u1, u2]
      return tf.group(u1, u2)

    # We only need to refresh the shortlist if n is greater than the
    # current shortlist size (which is stored in sl_ids[0]).
    with tf.control_dependencies(self.last_ops):
      cond_op = tf.cond(n > self.sl_ids[0], refresh_shortlist, tf.no_op)
      with tf.control_dependencies([cond_op]):
        topk_values, topk_indices = tf.nn.top_k(
            self.sl_scores, tf.minimum(n, tf.to_int32(self.sl_ids[0])))
        # topk_indices are the indices into the shortlist, we want to return
        # the indices into id_to_score
        gathered_indices = tf.gather(self.sl_ids, topk_indices)
        return gathered_indices, topk_values

  def get_and_remove_best(self, n):
    # TODO(thomaswc): Replace this with a version of get_best where
    # refresh_shortlist grabs the top n + shortlist_size.
    top_ids, unused_top_vals = self.get_best(n)
    remove_op = self.remove(top_ids)
    return tf.identity(top_ids, control_inputs=remove_op)
