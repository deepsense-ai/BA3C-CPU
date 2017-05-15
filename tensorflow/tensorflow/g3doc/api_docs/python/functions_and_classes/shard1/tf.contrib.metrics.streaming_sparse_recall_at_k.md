### `tf.contrib.metrics.streaming_sparse_recall_at_k(*args, **kwargs)` {#streaming_sparse_recall_at_k}

Computes recall@k of the predictions with respect to sparse labels. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

  If `class_id` is specified, we calculate recall by considering only the
      entries in the batch for which `class_id` is in the label, and computing
      the fraction of them for which `class_id` is in the top-k `predictions`.
  If `class_id` is not specified, we'll calculate recall as how often on
      average a class among the labels of a batch entry is in the top-k
      `predictions`.

  `streaming_sparse_recall_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_negative_at_<k>`, that are used to compute
  the recall_at_k frequency. This frequency is ultimately returned as
  `recall_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` +
  `false_negative_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false negatives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_negative_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match `predictions`.
      Values should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range always count
      towards `false_negative_at_<k>`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes), where num_classes is the last dimension of
      `predictions`. If class_id is outside this range, the method returns NAN.
    ignore_mask: An optional, `bool` `Tensor` whose shape is broadcastable to
      the the first [D1, ... DN] dimensions of `predictions` and `labels`.
    weights: An optional `Tensor` whose shape is broadcastable to the the first
      [D1, ... DN] dimensions of `predictions` and `labels`.
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately, and whose value matches
      `recall`.

  Raises:
    ValueError: If `ignore_mask` is not `None` and its shape doesn't match
      `predictions`, or if `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.

