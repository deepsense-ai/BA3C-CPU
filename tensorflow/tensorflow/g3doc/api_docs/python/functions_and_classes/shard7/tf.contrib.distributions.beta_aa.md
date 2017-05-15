Beta with softplus transform on `a` and `b`.
- - -

#### `tf.contrib.distributions.beta_aa.__init__(a, b, validate_args=False, allow_nan_stats=True, name='BetaWithSoftplusAB')` {#beta_aa.__init__}




- - -

#### `tf.contrib.distributions.beta_aa.a` {#beta_aa.a}

Shape parameter.


- - -

#### `tf.contrib.distributions.beta_aa.a_b_sum` {#beta_aa.a_b_sum}

Sum of parameters.


- - -

#### `tf.contrib.distributions.beta_aa.allow_nan_stats` {#beta_aa.allow_nan_stats}

Python boolean describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense.  E.g., the variance
of a Cauchy distribution is infinity.  However, sometimes the
statistic is undefined, e.g., if a distribution's pdf does not achieve a
maximum within the support of the distribution, the mode is undefined.
If the mean is undefined, then by definition the variance is undefined.
E.g. the mean for Student's T for df = 1 is undefined (no clear way to say
it is either + or - infinity), so the variance = E[(X - mean)^2] is also
undefined.

##### Returns:


*  <b>`allow_nan_stats`</b>: Python boolean.


- - -

#### `tf.contrib.distributions.beta_aa.b` {#beta_aa.b}

Shape parameter.


- - -

#### `tf.contrib.distributions.beta_aa.batch_shape(name='batch_shape')` {#beta_aa.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.beta_aa.cdf(value, name='cdf', **condition_kwargs)` {#beta_aa.cdf}

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
cdf(x) := P[X <= x]
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.beta_aa.dtype` {#beta_aa.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.beta_aa.entropy(name='entropy')` {#beta_aa.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.beta_aa.event_shape(name='event_shape')` {#beta_aa.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.beta_aa.get_batch_shape()` {#beta_aa.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.beta_aa.get_event_shape()` {#beta_aa.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.beta_aa.is_continuous` {#beta_aa.is_continuous}




- - -

#### `tf.contrib.distributions.beta_aa.is_reparameterized` {#beta_aa.is_reparameterized}




- - -

#### `tf.contrib.distributions.beta_aa.log_cdf(value, name='log_cdf', **condition_kwargs)` {#beta_aa.log_cdf}

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.


Additional documentation from `Beta`:

Note that the argument `x` must be a non-negative floating point tensor
whose shape can be broadcast with `self.a` and `self.b`.  For fixed leading
dimensions, the last dimension represents counts for the corresponding Beta
distribution in `self.a` and `self.b`. `x` is only legal if `0 < x < 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.beta_aa.log_pdf(value, name='log_pdf', **condition_kwargs)` {#beta_aa.log_pdf}

Log probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.beta_aa.log_pmf(value, name='log_pmf', **condition_kwargs)` {#beta_aa.log_pmf}

Log probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`log_pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.beta_aa.log_prob(value, name='log_prob', **condition_kwargs)` {#beta_aa.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.beta_aa.log_survival_function(value, name='log_survival_function', **condition_kwargs)` {#beta_aa.log_survival_function}

Log survival function.

Given random variable `X`, the survival function is defined:

```
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.beta_aa.mean(name='mean')` {#beta_aa.mean}

Mean.


- - -

#### `tf.contrib.distributions.beta_aa.mode(name='mode')` {#beta_aa.mode}

Mode.

Additional documentation from `Beta`:

Note that the mode for the Beta distribution is only defined
when `a > 1`, `b > 1`. This returns the mode when `a > 1` and `b > 1`,
and `NaN` otherwise. If `self.allow_nan_stats` is `False`, an exception
will be raised rather than returning `NaN`.


- - -

#### `tf.contrib.distributions.beta_aa.name` {#beta_aa.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.beta_aa.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#beta_aa.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.beta_aa.param_static_shapes(cls, sample_shape)` {#beta_aa.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.beta_aa.parameters` {#beta_aa.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.beta_aa.pdf(value, name='pdf', **condition_kwargs)` {#beta_aa.pdf}

Probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.beta_aa.pmf(value, name='pmf', **condition_kwargs)` {#beta_aa.pmf}

Probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.beta_aa.prob(value, name='prob', **condition_kwargs)` {#beta_aa.prob}

Probability density/mass function (depending on `is_continuous`).


Additional documentation from `Beta`:

Note that the argument `x` must be a non-negative floating point tensor
whose shape can be broadcast with `self.a` and `self.b`.  For fixed leading
dimensions, the last dimension represents counts for the corresponding Beta
distribution in `self.a` and `self.b`. `x` is only legal if `0 < x < 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.beta_aa.sample(sample_shape=(), seed=None, name='sample', **condition_kwargs)` {#beta_aa.sample}

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

##### Args:


*  <b>`sample_shape`</b>: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.


- - -

#### `tf.contrib.distributions.beta_aa.sample_n(n, seed=None, name='sample_n', **condition_kwargs)` {#beta_aa.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: `Scalar` `Tensor` of type `int32` or `int64`, the number of
    observations to sample.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with a prepended dimension (n,).

##### Raises:


*  <b>`TypeError`</b>: if `n` is not an integer type.


- - -

#### `tf.contrib.distributions.beta_aa.std(name='std')` {#beta_aa.std}

Standard deviation.


- - -

#### `tf.contrib.distributions.beta_aa.survival_function(value, name='survival_function', **condition_kwargs)` {#beta_aa.survival_function}

Survival function.

Given random variable `X`, the survival function is defined:

```
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.beta_aa.validate_args` {#beta_aa.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.beta_aa.variance(name='variance')` {#beta_aa.variance}

Variance.


