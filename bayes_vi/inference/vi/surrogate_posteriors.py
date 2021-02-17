import collections

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils import to_ordered_dict
from bayes_vi.utils.bijectors import CustomBlockwise

tfd = tfp.distributions
tfb = tfp.bijectors


class SurrogatePosterior:

    def __init__(self, model):
        self.model = model
        self.distribution = None
        self.reshape_sample_bijector = tfb.Chain([
            model.reshape_flat_param_bijector, model.split_flat_param_bijector
        ])
        self.unconstrained_event_ndims = model.flat_unconstrained_param_event_ndims
        self.event_ndims = model.flat_param_event_ndims
        dtypes = list(model.dtypes.values())
        if len(set(set(dtypes))) == 1:
            self.dtype = dtypes[0]
        else:
            raise ValueError('Model has incompatible dtypes: {}'.format(set(dtypes)))

    def approx_joint_marginal_posteriors(self, num_samples_to_approx_marginals):
        q = self.distribution.sample(num_samples_to_approx_marginals)
        if hasattr(self, 'extra_ndims') and self.extra_ndims > 0:
            q, a = tf.split(q, num_or_size_splits=[self.event_ndims, self.extra_ndims], axis=-1)
        posterior_samples = self.reshape_sample_bijector.forward(q)
        posteriors = self.model.get_param_distributions(param_samples=posterior_samples)
        return tfd.JointDistributionNamedAutoBatched(posteriors)


    def get_corrected_target_log_prob_fn(self, target_log_prob_fn):
        if hasattr(self, 'extra_ndims') and self.extra_ndims > 0:

            def corrected_target_log_prob_fn(sample):
                q, a = tf.split(
                    sample,
                    num_or_size_splits=[self.event_ndims, self.extra_ndims],
                    axis=-1
                )
                log_prob_q = target_log_prob_fn(self.reshape_sample_bijector.forward(q))
                log_prob_a = self.posterior_lift_distribution(
                    self.model.blockwise_constraining_bijector.inverse(q)).log_prob(a)
                return log_prob_q + log_prob_a
        else:

            def corrected_target_log_prob_fn(sample):
                return target_log_prob_fn(self.reshape_sample_bijector.forward(sample))

        return corrected_target_log_prob_fn


class ADVI(SurrogatePosterior):

    def __init__(self, model, mean_field=False):
        super(ADVI, self).__init__(model=model)

        loc = tf.Variable(tf.random.normal(shape=(self.unconstrained_event_ndims,), dtype=self.dtype), dtype=self.dtype)

        if mean_field:
            scale = tfp.util.TransformedVariable(
                tf.ones_like(loc),
                bijector=tfb.Softplus(),
            )
            self.distribution = tfd.TransformedDistribution(
                tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale),
                bijector=self.model.blockwise_constraining_bijector
            )

        else:
            scale_tril = tfp.util.TransformedVariable(
                tf.eye(self.unconstrained_event_ndims, dtype=self.dtype),
                bijector=tfb.FillScaleTriL(diag_bijector=tfb.Softplus(), diag_shift=1e-5),
                dtype=self.dtype
            )
            self.distribution = tfd.TransformedDistribution(
                tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril),
                bijector=self.model.blockwise_constraining_bijector
            )


class NormalizingFlow(SurrogatePosterior):

    def __init__(self, model, flow_bijector, extra_ndims=None, posterior_lift_distribution=None):
        super(NormalizingFlow, self).__init__(model=model)
        self.flow_bijector = flow_bijector

        if extra_ndims and extra_ndims < 0:
            raise ValueError('`extra_dims` have to be `None` or  `>=0`.')
        else:
            self.extra_ndims = extra_ndims if extra_ndims else 0

        loc = tf.zeros(shape=(self.unconstrained_event_ndims + self.extra_ndims,), dtype=self.dtype)

        scale = tf.ones_like(loc)

        base_dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        transformed = tfd.TransformedDistribution(
            distribution=base_dist, bijector=self.flow_bijector
        )

        if self.extra_ndims > 0:
            blockwise_constraining_bijector = tfb.Chain([
                tfb.Invert(tfb.Split([self.event_ndims, self.extra_ndims])),
                tfb.JointMap(bijectors=[self.model.blockwise_constraining_bijector, tfb.Identity()]),
                tfb.Split([self.unconstrained_event_ndims, self.extra_ndims])
            ])
        else:
            blockwise_constraining_bijector = self.model.blockwise_constraining_bijector

        self.distribution = tfd.TransformedDistribution(
            distribution=transformed, bijector=blockwise_constraining_bijector
        )

        self.posterior_lift_distribution = posterior_lift_distribution
        if self.extra_ndims > 0 and not self.posterior_lift_distribution:
            self.posterior_lift_distribution = lambda _: tfd.MultivariateNormalDiag(
                loc=tf.zeros(shape=(self.extra_ndims,), dtype=self.dtype),
                scale_diag=tf.ones(shape=(self.extra_ndims,), dtype=self.dtype),
            )
