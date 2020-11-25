import collections
import functools

import tensorflow_probability as tfp


def make_transform_fn(bijector, direction):
    """Makes a function which applies a list of Bijectors.

    Note: Adapted from tensorflow_probability.python.mcmc.transformed_kernel

    Parameters
    ----------
    bijector: `tfp.bijectors.Bijector` or `list` or `tfp.bijectors.Bijector`
        A bijector or list of bijectors to apply.
    direction: {'forward', 'inverse'}
        Direction in which to apply bijectors.

    Returns
    -------
    `callable`
        A function taking a `list` of `tf.Tensor`, which applies the list of bijectors.
    """
    if not isinstance(bijector, (tuple, list)):
        bijector = [bijector]

    def fn(state_parts):
        return [getattr(b, direction)(sp) for b, sp in zip(bijector, state_parts)]

    return fn


def make_val_and_grad_fn(value_fn):
    """Decorator to transform a function to return value and gradient."""

    @functools.wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)

    return val_and_grad


def compose(fns):
    """Creates a function composition."""

    def composition(*args, fns_):
        res = fns_[0](*args)
        for f in fns_[1:]:
            res = f(*res)
        return res

    return functools.partial(composition, fns_=fns)


def to_ordered_dict(param_names, state):
    """Transforms list of state parts into collections.OrderedDict with given param names."""
    return collections.OrderedDict([(k, v) for k, v in zip(param_names, state)])
