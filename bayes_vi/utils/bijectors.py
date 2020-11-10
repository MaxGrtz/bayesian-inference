import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

tfd = tfp.distributions
tfb = tfp.bijectors


class Mapper:
    """Basically, this is a bijector without log-jacobian correction."""

    def __init__(self, list_of_tensors, list_of_bijectors, event_shape):
        self.dtype = dtype_util.common_dtype(
            list_of_tensors, dtype_hint=tf.float32)
        self.list_of_tensors = list_of_tensors
        self.bijectors = list_of_bijectors
        self.event_shape = event_shape

    def flatten_and_concat(self, list_of_tensors):
        def _reshape_map_part(part, event_shape, bijector):
            part = tf.cast(bijector.inverse(part), self.dtype)
            static_rank = tf.get_static_value(ps.rank_from_shape(event_shape))
            if static_rank == 1:
                return part
            new_shape = ps.concat([
                ps.shape(part)[:ps.size(ps.shape(part)) - ps.size(event_shape)],
                [-1]
            ], axis=-1)
            return tf.reshape(part, ps.cast(new_shape, tf.int32))

        x = tf.nest.map_structure(_reshape_map_part,
                                  list_of_tensors,
                                  self.event_shape,
                                  self.bijectors)
        return tf.concat(tf.nest.flatten(x), axis=-1)

    def split_and_reshape(self, x):
        assertions = []
        message = 'Input must have at least one dimension.'
        if tensorshape_util.rank(x.shape) is not None:
            if tensorshape_util.rank(x.shape) == 0:
                raise ValueError(message)
        else:
            assertions.append(assert_util.assert_rank_at_least(x, 1, message=message))
        with tf.control_dependencies(assertions):
            splits = [
                tf.cast(ps.maximum(1, ps.reduce_prod(s)), tf.int32)
                for s in tf.nest.flatten(self.event_shape)
            ]
            x = tf.nest.pack_sequence_as(
                self.event_shape, tf.split(x, splits, axis=-1))

            def _reshape_map_part(part, part_org, event_shape, bijector):
                part = tf.cast(bijector.forward(part), part_org.dtype)
                static_rank = tf.get_static_value(ps.rank_from_shape(event_shape))
                if static_rank == 1:
                    return part
                new_shape = ps.concat([ps.shape(part)[:-1], event_shape], axis=-1)
                return tf.reshape(part, ps.cast(new_shape, tf.int32))

            x = tf.nest.map_structure(_reshape_map_part,
                                      x,
                                      self.list_of_tensors,
                                      self.event_shape,
                                      self.bijectors)
        return x
