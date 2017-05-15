Bijector which inverts another Bijector.

Example Use: [ExpGammaDistribution (see Background & Context)](
https://reference.wolfram.com/language/ref/ExpGammaDistribution.html)
models `Y=log(X)` where `X ~ Gamma`.

```python
exp_gamma_distribution = TransformedDistribution(
  Gamma(alpha=1., beta=2.),
  bijector.Invert(bijector.Exp())
```
- - -

#### `tf.contrib.distributions.bijector.Invert.__init__(bijector, validate_args=False, name=None)` {#Invert.__init__}

Creates a `Bijector` which swaps the meaning of `inverse` and `forward`.

Note: An inverted bijector's `inverse_log_det_jacobian` is often more
efficient if the base bijector implements `_forward_log_det_jacobian`. If
`_forward_log_det_jacobian` is not implemented then the following code is
used:

```python
y = self.inverse(x, **condition_kwargs)
return -self.inverse_log_det_jacobian(y, **condition_kwargs)
```

##### Args:


*  <b>`bijector`</b>: Bijector instance.
*  <b>`validate_args`</b>: `Boolean` indicated whether arguments should be checked for
    correctness.
*  <b>`name`</b>: `String`, name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Invert.bijector` {#Invert.bijector}




- - -

#### `tf.contrib.distributions.bijector.Invert.dtype` {#Invert.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward(x, name='forward', **condition_kwargs)` {#Invert.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Invert.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse(y, name='inverse', **condition_kwargs)` {#Invert.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Invert.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Invert.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.is_constant_jacobian` {#Invert.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Invert.name` {#Invert.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Invert.parameters` {#Invert.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Invert.shaper` {#Invert.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Invert.validate_args` {#Invert.validate_args}

Returns True if Tensor arguments will be validated.


