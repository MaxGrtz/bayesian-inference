import collections

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils import to_ordered_dict
from bayes_vi.utils.bijectors import CustomBlockwise
from bayes_vi.inference.vi.flow_bijectors import make_shift_fn, make_scale_fn

tfd = tfp.distributions
tfb = tfp.bijectors


class SurrogatePosterior:

    def __init__(self, model):
        self.model = model
        self.posterior_distribution = None

    def approx_joint_marginal_posteriors(self, num_samples_to_approx_marginals):
        reshaped_samples = self.reshape_sample(self.posterior_distribution.sample(num_samples_to_approx_marginals))
        posteriors = collections.OrderedDict(
            [(name, tfd.Empirical(tf.reshape(part, shape=(-1, *list(event_shape))), event_ndims=len(event_shape)))
             for (name, part), event_shape
             in zip(reshaped_samples.items(), self.model.prior_distribution.event_shape.values())]
        )
        return tfd.JointDistributionNamedAutoBatched(posteriors)

    def posterior_stats(self, num_samples=10000):
        joint_marginals = self.approx_joint_marginal_posteriors()
        samples = joint_marginals.sample(num_samples)
        # compute stats
        raise NotImplementedError('Posterior statistics computation is not yet implemented!')


    def reshape_sample(self, sample):
        return to_ordered_dict(
            self.model.param_names,
            self.model.reshape_flat_constrained_sample(
                self.model.split_constrained_bijector.forward(sample)
            )
        )

    def unconstrain_flatten_and_merge(self, sample):
        return self.model.split_unconstrained_bijector.inverse(
            self.model.flatten_unconstrained_sample(
                self.model.unconstrain_sample(sample.values())
            )
        )

    def get_target_log_prob_fn(self, target_log_prob):
        return lambda sample: target_log_prob(
            self.reshape_sample(sample)
        )


class ADVI(SurrogatePosterior):

    def __init__(self, model, mean_field=False):
        super(ADVI, self).__init__(model=model)

        sample = self.unconstrain_flatten_and_merge(self.model.prior_distribution.sample())

        loc = tf.Variable(
            tf.random.normal(sample.shape, dtype=sample.dtype),
            name='meanfield_mu',
            dtype=sample.dtype)

        if mean_field:
            scale = tfp.util.TransformedVariable(
                tf.fill(sample.shape, value=tf.constant(0.5, sample.dtype)),
                tfb.Softplus(),
                name='meanfield_scale',
            )
            self.posterior_distribution = tfd.TransformedDistribution(
                tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale),
                bijector=self.model.blockwise_constraining_bijector
            )

        else:
            bij = tfb.Chain([
                tfb.TransformDiagonal(tfb.Softplus()),
                tfb.FillTriangular()])
            scale_tril = tfp.util.TransformedVariable(
                tf.linalg.diag(tf.fill(sample.shape, value=tf.constant(0.5, sample.dtype))),
                bijector=bij,
                name='meanfield_scale',
            )
            self.posterior_distribution = tfd.TransformedDistribution(
                tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril),
                bijector=model.blockwise_constraining_bijector
            )


class NormalizingFlow(SurrogatePosterior):

    def __init__(self, model, flow_bijector):
        super(NormalizingFlow, self).__init__(model=model)
        self.flow_bijector = flow_bijector

        sample = self.unconstrain_flatten_and_merge(self.model.prior_distribution.sample())

        loc = tf.zeros_like(sample)

        scale = tf.ones(shape=sample.shape, dtype=sample.dtype)

        base_dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        transformed = tfd.TransformedDistribution(
            distribution=base_dist, bijector=self.flow_bijector
        )

        self.posterior_distribution = tfd.TransformedDistribution(
            distribution=transformed, bijector=self.model.blockwise_constraining_bijector
        )


class AugmentedNormalizingFlow(SurrogatePosterior):

    def __init__(self, model, flow_bijector, extra_dims=None, posterior_lift_distribution=None):
        super(AugmentedNormalizingFlow, self).__init__(model=model)

        if extra_dims and extra_dims <= 0:
            raise ValueError('`extra_dims` have to be `None` or  `> 0`.')

        self.flow_bijector = flow_bijector
        sample = self.unconstrain_flatten_and_merge(self.model.prior_distribution.sample())
        self.dims = sample.shape[0]
        if not extra_dims:
            self.extra_dims = self.dims
        else:
            self.extra_dims = extra_dims

        loc = tf.zeros(shape=(self.dims + self.extra_dims), dtype=sample.dtype)

        scale = tf.ones(shape=(self.dims + self.extra_dims), dtype=sample.dtype)

        base_dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        transformed = tfd.TransformedDistribution(
            distribution=base_dist, bijector=self.flow_bijector
        )

        self.posterior_lift_distribution = posterior_lift_distribution
        if not self.posterior_lift_distribution:
            self.posterior_lift_distribution = lambda _: tfd.MultivariateNormalDiag(
                loc=tf.zeros_like(sample),
                scale_identity_multiplier=1.,
            )

        constraining_bijector = CustomBlockwise(
            input_block_sizes=[self.dims, self.extra_dims],
            output_block_sizes=[self.model.blockwise_constraining_bijector.output_event_shape[0], self.extra_dims],
            bijectors=[self.model.blockwise_constraining_bijector, tfb.Identity()]
        )

        self.posterior_distribution = tfd.TransformedDistribution(
            distribution=transformed, bijector=constraining_bijector
        )

    def get_target_log_prob_fn(self, target_log_prob):

        def target_log_prob_fn(sample):
            q, a = tf.split(sample, num_or_size_splits=[sample.shape[-1] - self.extra_dims, self.extra_dims], axis=-1)
            log_prob_q = target_log_prob(self.reshape_sample(q))
            log_prob_a = self.posterior_lift_distribution(self.model.blockwise_constraining_bijector.inverse(q)).log_prob(a)
            return log_prob_q + log_prob_a

        return target_log_prob_fn

    def approx_joint_marginal_posteriors(self, num_samples_to_approx_marginals):
        samples = self.posterior_distribution.sample(num_samples_to_approx_marginals)
        q, a = tf.split(samples, num_or_size_splits=[samples.shape[-1] - self.extra_dims, self.extra_dims], axis=-1)
        reshaped_samples = self.reshape_sample(q)
        posteriors = collections.OrderedDict(
            [(name, tfd.Empirical(tf.reshape(part, shape=(-1, *list(event_shape))), event_ndims=len(event_shape)))
             for (name, part), event_shape
             in zip(reshaped_samples.items(), self.model.prior_distribution.event_shape.values())]
        )
        return tfd.JointDistributionNamedAutoBatched(posteriors)
