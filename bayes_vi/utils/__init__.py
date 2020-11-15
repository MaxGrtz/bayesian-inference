import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static


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


def make_log_det_jacobian_fn(bijector, direction):
    """Makes a function which applies a list of Bijectors' `log_det_jacobian`s.

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
    attr = '{}_log_det_jacobian'.format(direction)

    if not isinstance(bijector, (tuple, list)):
        dtype = getattr(bijector, '{}_dtype'.format(direction))()

        if isinstance(dtype, (tuple, list)):
            def multipart_fn(state_parts, event_ndims):
                return getattr(bijector, attr)(state_parts, event_ndims)

            return multipart_fn
        elif tf.nest.is_nested(dtype):
            raise ValueError(
                'Only list-like multi-part bijectors are currently supported, but '
                'got {}.'.format(tf.nest.map_structure(lambda _: '.', dtype)))

        bijector = [bijector]

    def fn(state_parts, event_ndims):
        return sum([
            getattr(b, attr)(sp, event_ndims=e)
            for b, e, sp in zip(bijector, event_ndims, state_parts)
        ])

    return fn


def make_transformed_log_prob(log_prob_fn, bijector, direction,
                              targets_fixed=False, split_bijector=None, enable_bijector_caching=True):
    """Transforms a log_prob function using bijectors.

    Note: Adapted from tensorflow_probability.python.mcmc.transformed_kernel

    Parameters
    ----------
    log_prob_fn: `callable`
        taking an argument for each state part and
        returns a `Tensor` representing the joint `log` probability of those state parts.
    bijector: `tfp.bijectors.Bijector`-like instance (or `list` thereof)
        corresponding to each state part. When `direction = 'forward'` the
        `Bijector`-like instance must possess members `forward` and
        `forward_log_det_jacobian` (and corresponding when
        `direction = 'inverse'`).
    direction: {'forward', 'inverse'}
        Direction of the bijector transformation applied to each state part.
    enable_bijector_caching: `bool`
        Indicates if `Bijector` caching should be invalidated.
        Default value: `True`.

    Returns
    -------
    transformed_log_prob_fn: `callable`
        which takes an argument for each transformed state part and
        returns a `Tensor` representing the joint `log` probability of the transformed state parts.
    """
    if direction not in {'forward', 'inverse'}:
        raise ValueError('Argument `direction` must be either `"forward"` or `"inverse"`; saw "{}".'.format(direction))
    fn = make_transform_fn(bijector, direction)
    ldj_fn = make_log_det_jacobian_fn(bijector, direction)

    if targets_fixed:
        def transformed_log_prob_fn(*state_parts):
            """Log prob of the transformed state."""
            if not enable_bijector_caching:
                state_parts = [tf.identity(sp) for sp in state_parts]
            if split_bijector is not None:
                state_parts = split_bijector(state_parts[0])
            tlp = log_prob_fn(*fn(state_parts))
            tlp_rank = prefer_static.rank(tlp)
            event_ndims = [(prefer_static.rank(sp) - tlp_rank) for sp in state_parts]
            ldj = ldj_fn(state_parts, event_ndims)
            try:
                return tlp + tf.reduce_sum(ldj, axis=0)
            except ValueError:
                return tlp + ldj
    else:
        def transformed_log_prob_fn(state_parts, y):
            """Log prob of the transformed state."""
            if not enable_bijector_caching:
                state_parts = [tf.identity(sp) for sp in state_parts]
            if split_bijector is not None:
                state_parts = split_bijector(state_parts[0])
            tlp = log_prob_fn(fn(state_parts), y)
            tlp_rank = prefer_static.rank(tlp)
            event_ndims = [(prefer_static.rank(sp) - tlp_rank) for sp in state_parts]
            ldj = ldj_fn(state_parts, event_ndims)
            try:
                return tlp + tf.reduce_sum(ldj, axis=0)
            except ValueError:
                return tlp + ldj

    return transformed_log_prob_fn


def make_val_and_grad_fn(value_fn):
    """Decorator to transform a function to return value and gradient."""

    @functools.wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)

    return val_and_grad


def to_ordered_dict(param_names, state):
    """Transforms list of state parts into collections.OrderedDict with given param names."""
    return collections.OrderedDict([(k, v) for k, v in zip(param_names, state)])
