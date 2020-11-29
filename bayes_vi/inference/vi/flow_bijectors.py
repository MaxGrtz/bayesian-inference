import tensorflow_probability as tfp
import tensorflow as tf
from bayes_vi.utils.leapfrog_integrator import LeapfrogIntegrator

tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors


class AffineFlow(tfb.Bijector):

    def __init__(self, event_shape, validate_args=False, name='affine_flow'):
        super(AffineFlow, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name
        )

        self.event_shape = event_shape

        bij = tfb.Chain([
            tfb.TransformDiagonal(tfb.Softplus()),
            tfb.FillTriangular()
        ])

        self.shift = tf.Variable(tf.random.normal(shape=self.event_shape))

        self.scale_tril = tfp.util.TransformedVariable(
            tf.linalg.diag(tf.fill(self.event_shape, value=tf.constant(0.5))),
            bijector=bij,
        )

        self.bijector = tfb.Chain([
            tfb.Shift(self.shift),
            tfb.ScaleMatvecTriL(self.scale_tril)
        ])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def forward_log_det_jacobian(self, x, event_ndims, name='forward_log_det_jacobian', **kwargs):
        return self.bijector.forward_log_det_jacobian(x, event_ndims)

    def inverse_log_det_jacobian(self, y, event_ndims, name='inverse_log_det_jacobian', **kwargs):
        return self.bijector.inverse_log_det_jacobian(y, event_ndims)


    @classmethod
    def _is_increasing(cls, **kwargs):
        return False


class PhaseSpaceTransform(tfb.Bijector):

    def __init__(self, event_shape, shift_fn=None, scale_fn=None, hidden_layers=None,
                 validate_args=False, name='phase_space_transform'):
        super(PhaseSpaceTransform, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            is_constant_jacobian=True,
            name=name
        )
        self.shift_fn = shift_fn
        self.scale_fn = scale_fn
        if self.shift_fn is None:
            self.shift_fn = make_shift_fn(event_shape=event_shape, hidden_layers=hidden_layers)
        if self.scale_fn is None:
            self.scale_fn = make_scale_fn(event_shape=event_shape, hidden_layers=hidden_layers)

    def _forward(self, x):
        p = self.scale_fn(x) * tf.random.normal(shape=tf.shape(x)) + self.shift_fn(x)
        return tf.concat([x, p], axis=-1)

    def _inverse(self, y):
        x, p = tf.split(y, num_or_size_splits=2, axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype)

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0., y.dtype)

    def _forward_event_shape(self, input_shape):
        shape_list = list(input_shape)
        shape_list[-1] = shape_list[-1] * 2
        return tf.TensorShape(shape_list)

    def _inverse_event_shape(self, output_shape):
        shape_list = list(output_shape)
        shape_list[-1] = int(shape_list[-1] / 2)
        return tf.TensorShape(shape_list)

    def _forward_event_shape_tensor(self, input_shape):
        shape, last = tf.split(input_shape, num_or_size_splits=[tf.rank(input_shape)-1, 1], axis=-1)
        new_last = last * 2
        output_shape = tf.concat([shape, new_last], axis=-1)
        return output_shape

    def _inverse_event_shape_tensor(self, output_shape):
        shape, last = tf.split(output_shape, num_or_size_splits=[tf.rank(output_shape)-1, 1], axis=-1)
        new_last = tf.cast(last / 2, dtype=tf.int32)
        input_shape = tf.concat([shape, new_last], axis=-1)
        return input_shape

    @classmethod
    def _is_increasing(cls, **kwargs):
        return False


class HamiltonianFlow(tfb.Bijector):

    def __init__(self, event_shape, hamiltonian_fn=None, symplectic_integrator=LeapfrogIntegrator(),
                 step_sizes=None, num_leapfrog_steps=3, hidden_layers=None,
                 validate_args=False, name='hamiltonian_flow'):
        super(HamiltonianFlow, self).__init__(is_constant_jacobian=True,
                                              validate_args=validate_args,
                                              forward_min_event_ndims=1,
                                              inverse_min_event_ndims=1,
                                              name=name)

        self.hamiltonian_fn = hamiltonian_fn
        if hamiltonian_fn is None:
            self.hamiltonian_fn = make_hamiltonian_fn(event_shape=event_shape, hidden_layers=hidden_layers)
        self.symplectic_integrator = symplectic_integrator
        self.step_sizes = step_sizes
        if self.step_sizes is None:
            self.step_sizes = tf.Variable(tf.ones(event_shape)*0.1e-2)
        self.num_leapfrog_steps = num_leapfrog_steps

    def _forward(self, x):
        return self.symplectic_integrator.solve(self.hamiltonian_fn, initial_state=x, step_sizes=self.step_sizes,
                                                num_leapfrog_steps=self.num_leapfrog_steps)

    def _inverse(self, y):
        return self.symplectic_integrator.solve(self.hamiltonian_fn, initial_state=y, step_sizes=-self.step_sizes,
                                                num_leapfrog_steps=self.num_leapfrog_steps)

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype)

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0., y.dtype)

    @classmethod
    def _is_increasing(cls, **kwargs):
        return False


def make_hamiltonian_fn(event_shape, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [32, 32]
    hamiltonian_fn = tfk.Sequential()
    for n in hidden_layers:
        hamiltonian_fn.add(tfk.layers.Dense(n, activation=tf.keras.activations.tanh))
    hamiltonian_fn.add(tfk.layers.Dense(1, activation=tf.keras.activations.softplus))
    hamiltonian_fn.build((None,) + (event_shape[0] * 2,))
    return hamiltonian_fn


def make_scale_fn(event_shape, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [32]
    scale_fn = tfk.Sequential()
    for n in hidden_layers:
        scale_fn.add(tfk.layers.Dense(n, activation=tf.keras.activations.relu(max_value=6)))
    scale_fn.add(tfk.layers.Dense(event_shape[0], activation=tf.keras.activations.softplus))
    scale_fn.build((None,) + tuple(event_shape))
    return scale_fn


def make_shift_fn(event_shape, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [32]
    shift_fn = tfk.Sequential()
    for n in hidden_layers:
        shift_fn.add(tfk.layers.Dense(n, activation=tf.keras.activations.relu(max_value=6)))
    shift_fn.add(tfk.layers.Dense(event_shape[0], activation=tf.keras.activations.linear))
    shift_fn.build((None,) + tuple(event_shape))
    return shift_fn
