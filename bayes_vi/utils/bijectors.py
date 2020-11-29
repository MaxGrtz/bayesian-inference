import tensorflow_probability as tfp
import tensorflow as tf


from bayes_vi.utils import make_transform_fn

tfb = tfp.bijectors


class CustomBlockwise(tfb.Bijector):
    """Custom modification of tfp.bijectors.Blockwise."""

    def __init__(self, bijectors, input_block_sizes, output_block_sizes,
                 validate_args=False, name='custom_blockwise'):
        super(CustomBlockwise, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            name=name
        )
        self.bijectors = bijectors
        self.input_event_shape = [sum(input_block_sizes)]
        self.output_event_shape = [sum(output_block_sizes)]

        self.input_split_bijector = tfb.Split(num_or_size_splits=input_block_sizes)
        self.output_split_bijector = tfb.Split(num_or_size_splits=output_block_sizes)

        self.forward_fn = make_transform_fn(bijector=bijectors, direction='forward')
        self.inverse_fn = make_transform_fn(bijector=bijectors, direction='inverse')

    def _forward(self, x):
        return self.output_split_bijector.inverse(
            self.forward_fn(self.input_split_bijector.forward(x))
        )

    def _inverse(self, y):
        return self.input_split_bijector.inverse(
            self.inverse_fn(self.output_split_bijector.forward(y))
        )

    def forward_log_det_jacobian(self, x, event_ndims, name='forward_log_det_jacobian', **kwargs):
        return self._forward_log_det_jacobian(x)

    def _forward_log_det_jacobian(self, x):
        return sum(
            b.forward_log_det_jacobian(x_, event_ndims=1)
            for b, x_ in zip(self.bijectors, self.input_split_bijector.forward(x))
        )

    def inverse_log_det_jacobian(self, y, event_ndims, name='inverse_log_det_jacobian', **kwargs):
        return self._inverse_log_det_jacobian(y)

    def _inverse_log_det_jacobian(self, y):
        return sum(
            [b.inverse_log_det_jacobian(y_, event_ndims=1)
             for b, y_ in zip(self.bijectors, self.output_split_bijector.forward(y))]
        )

    def _forward_event_shape(self, input_shape):
        return tf.TensorShape(self.output_event_shape)

    def _inverse_event_shape(self, output_shape):
        return tf.TensorShape(self.input_event_shape)

    def _forward_event_shape_tensor(self, input_shape):
        return tf.constant(self.output_event_shape, dtype=tf.int32)

    def _inverse_event_shape_tensor(self, output_shape):
        return tf.constant(self.input_event_shape, dtype=tf.int32)

    @classmethod
    def _is_increasing(cls, **kwargs):
        return False
