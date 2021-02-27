import functools

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils.leapfrog_integrator import LeapfrogIntegrator

tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors


class AffineFlow(tfb.Bijector):
    """Implements a trainable Affine Flow bijector."""

    def __init__(self, event_dims, validate_args=False, name='affine_flow'):
        super(AffineFlow, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name
        )

        bij = tfb.Chain([
            tfb.TransformDiagonal(tfb.Softplus()),
            tfb.FillTriangular()
        ])

        self.shift = tf.Variable(tf.random.normal(shape=[event_dims]))

        self.scale_tril = tfp.util.TransformedVariable(
            tf.linalg.diag(tf.fill([event_dims], value=tf.constant(0.5))),
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


class HamiltonianFlow(tfb.Bijector):
    """Implements a trainable Hamiltonian Flow bijector."""

    def __init__(self, event_dims, potential_energy_fn=None, kinetic_energy_fn=None,
                 symplectic_integrator=LeapfrogIntegrator(), step_sizes=0.2, num_integration_steps=5,
                 hidden_layers=None, validate_args=False, name='hamiltonian_flow'):
        super(HamiltonianFlow, self).__init__(is_constant_jacobian=True,
                                              validate_args=validate_args,
                                              forward_min_event_ndims=1,
                                              inverse_min_event_ndims=1,
                                              name=name)

        self.potential_energy_fn = potential_energy_fn
        if potential_energy_fn is None:
            self.potential_energy_fn = make_energy_fn(event_dims=event_dims, hidden_layers=hidden_layers)

        self.kinetic_energy_fn = kinetic_energy_fn
        if kinetic_energy_fn is None:
            self.kinetic_energy_fn = make_energy_fn(event_dims=event_dims, hidden_layers=hidden_layers)

        self.symplectic_integrator = symplectic_integrator
        self.num_integration_steps = num_integration_steps
        self.step_sizes = step_sizes

    def _forward(self, x):
        return self.symplectic_integrator.solve(self.potential_energy_fn, self.kinetic_energy_fn, initial_state=x,
                                                step_sizes=self.step_sizes,
                                                num_integration_steps=self.num_integration_steps)

    def _inverse(self, y):
        return self.symplectic_integrator.solve(self.potential_energy_fn, self.kinetic_energy_fn, initial_state=y,
                                                step_sizes=-self.step_sizes,
                                                num_integration_steps=self.num_integration_steps)

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype)

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0., y.dtype)

    @classmethod
    def _is_increasing(cls, **kwargs):
        return False


def make_energy_fn(event_dims, hidden_layers=None):
    """Utility function to construct a simple MLP energy function."""
    if hidden_layers is None:
        hidden_layers = [32, 32]
    energy_fn = tfk.Sequential()
    for n in hidden_layers:
        energy_fn.add(tfk.layers.Dense(n, activation=tf.keras.activations.tanh))
    energy_fn.add(tfk.layers.Dense(1, activation=tf.keras.activations.softplus))
    energy_fn.build((None,) + tuple([event_dims]))
    return energy_fn


def make_scale_fn(event_ndims, hidden_layers=None):
    """Utility function to construct a simple MLP function to parameterize a diagonal scale."""
    if hidden_layers is None:
        hidden_layers = [32]
    scale_fn = tfk.Sequential()
    for n in hidden_layers:
        scale_fn.add(tfk.layers.Dense(n, activation=functools.partial(tf.keras.activations.relu, max_value=6)))
    scale_fn.add(tfk.layers.Dense(event_ndims, activation=tf.keras.activations.softplus))
    scale_fn.build((None,) + (event_ndims,))
    return scale_fn


def make_shift_fn(event_ndims, hidden_layers=None):
    """Utility function to construct a simple MLP shift function to parameterize a location."""
    if hidden_layers is None:
        hidden_layers = [32]
    shift_fn = tfk.Sequential()
    for n in hidden_layers:
        shift_fn.add(tfk.layers.Dense(n, activation=functools.partial(tf.keras.activations.relu, max_value=6)))
    shift_fn.add(tfk.layers.Dense(event_ndims, activation=tf.keras.activations.linear))
    shift_fn.build((None,) + (event_ndims,))
    return shift_fn
