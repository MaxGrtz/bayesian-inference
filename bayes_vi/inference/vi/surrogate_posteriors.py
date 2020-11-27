import tensorflow as tf
import tensorflow_probability as tfp
import collections

from bayes_vi.utils import to_ordered_dict


tfd = tfp.distributions
tfb = tfp.bijectors


class SurrogatePosterior:

    def __init__(self, model):
        self.model = model
        self.posterior_distribution = None
        self.posteriors = None
        self.joint_marginal_posteriors = None

    def approx_joint_marginal_posteriors(self, samples_to_approx_marginals):
        reshaped_samples = self.reshape_sample(self.posterior_distribution.sample(samples_to_approx_marginals))

        self.posteriors = collections.OrderedDict(
            [(name, tfd.Empirical(tf.reshape(part, shape=(-1, *list(event_shape))), event_ndims=len(event_shape)))
             for (name, part), event_shape
             in zip(reshaped_samples.items(), self.model.prior_distribution.event_shape.values())]
        )

        self.joint_marginal_posteriors = tfd.JointDistributionNamedAutoBatched(self.posteriors)
        return self.joint_marginal_posteriors

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


class MeanFieldADVI(SurrogatePosterior):


    def __init__(self, model):
        super(MeanFieldADVI, self).__init__(model=model)

        sample = self.unconstrain_flatten_and_merge(self.model.prior_distribution.sample())

        loc = tf.Variable(
            tf.random.normal(sample.shape, dtype=sample.dtype),
            name='meanfield_mu',
            dtype=sample.dtype)

        scale = tfp.util.TransformedVariable(
            tf.fill(sample.shape, value=tf.constant(0.5, sample.dtype)),
            tfb.Softplus(),
            name='meanfield_scale',
        )

        self.posterior_distribution = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale),
            bijector=self.model.blockwise_constraining_bijector
        )


class ADVI(SurrogatePosterior):

    def __init__(self, model):
        super(ADVI, self).__init__(model=model)

        sample = self.unconstrain_flatten_and_merge(self.model.prior_distribution.sample())

        loc = tf.Variable(
            tf.random.normal(sample.shape, dtype=sample.dtype),
            name='meanfield_mu',
            dtype=sample.dtype)

        bij = tfb.Chain([
                tfb.TransformDiagonal(tfb.Shift(tf.ones(shape=sample.shape, dtype=tf.float32) * 1e-1)),
                tfb.TransformDiagonal(tfb.Exp()),
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

        loc = tf.random.normal(sample.shape, dtype=sample.dtype)

        scale = tf.fill(sample.shape, value=tf.constant(0.5, sample.dtype))

        base_dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        transformed = tfd.TransformedDistribution(
            distribution=base_dist, bijector=self.flow_bijector
        )

        self.posterior_distribution = tfd.TransformedDistribution(
            distribution=transformed, bijector=self.model.blockwise_constraining_bijector
        )
