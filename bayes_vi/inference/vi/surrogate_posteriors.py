"""Module providing different surrogate posterior classes."""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class SurrogatePosterior:
    """Base Class for surrogate posteriors.

    Attributes
    ----------
    model: `Model`
        A Bayesian probabilistic `Model`.
    distribution: `tfp.distributions.Distribution`
        The trainable surrogate posterior distribution.
    reshape_sample_bijector: `tfp.bijectors.Bijector`
        A bijector to split merged parameters and reshape them.
    unconstrained_event_ndims: `int`
        Number of unconstrained parameter space dimensions.
    event_ndims: `int`
        Number of parameter space dimensions.
    dtype: `tf.dtypes.Dtype`
        Datatype of the surrogate posterior distribution samples.
    """

    def __init__(self, model):
        """Initializes surrogate posterior instance.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        """

        self.model = model
        self.distribution = None
        self.reshape_sample_bijector = tfb.Chain([
            self.model.reshape_flat_param_bijector, self.model.split_flat_param_bijector
        ])
        self.unconstrained_event_ndims = self.model.flat_unconstrained_param_event_ndims
        self.event_ndims = self.model.flat_param_event_ndims
        dtypes = list(self.model.dtypes.values())
        if len(set(set(dtypes))) == 1:
            self.dtype = dtypes[0]
        else:
            raise ValueError('Model has incompatible dtypes: {}'.format(set(dtypes)))

    def approx_joint_marginal_posteriors(self, num_samples_to_approx_marginals):
        """Approximates the marginal posterior distributions and returns them as a joint distribution.

        Parameters
        ----------
        num_samples_to_approx_marginals: `int`
            The number of samples to approximate the marginal posteriors for each parameter.

        Returns
        -------
        `tfp.distributions.JointDistributionNamedAutoBatched`
            The independent joint distribution over model parameters.
        """

        q = self.distribution.sample(num_samples_to_approx_marginals)
        if hasattr(self, 'extra_ndims') and self.extra_ndims > 0:
            q, a = tf.split(q, num_or_size_splits=[self.event_ndims, self.extra_ndims], axis=-1)
        posterior_samples = self.reshape_sample_bijector.forward(q)
        posteriors = self.model.get_param_distributions(param_samples=posterior_samples)
        return tfd.JointDistributionNamedAutoBatched(posteriors)

    def get_corrected_target_log_prob_fn(self, target_log_prob_fn):
        """If the surrogate posterior has an Attribute `extra_ndims`, return the corrected `target_log_prob_fn`.


        """
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
    """Implements Automatic Differentiation Variational Inference (ADVI) surrogate posterior."""

    def __init__(self, model, mean_field=False):
        """Initializes ADVI surrogate posterior.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        mean_field: `bool`
            A boolean indicator whether or not to use meanfield ADVI. (Default: `False`)
        """

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
    """Implements Normalizing Flow surrogate posterior.

    Attributes
    ----------
    base_distribution: `tfp.distributions.Distribution`
        The base distribution of the normalizing flow based surrogate posterior.
    flow_bijector: `tfp.bijectors.Bijector`
        A trainable bijector.
    extra_ndims: `int`
        Number of extra dimensions for augmented normalizing flows.
    posterior_lift_distribution: `callable`
        A callable returning a `tfp.distributions.Distribution`, implementing a
        conditional lift distribution over extra dimensions for augmented normalizing flows.
    """

    def __init__(self, model, flow_bijector, base_distribution=None, extra_ndims=None, posterior_lift_distribution=None):
        """Initializes NormalizingFlow surrogate posterior.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        flow_bijector: `tfp.bijectors.Bijector`
            A trainable bijector.
        base_distribution: `tfp.distributions.Distribution`
            The base distribution of the normalizing flow based surrogate posterior.
            (Default: tfp.distributions.MultivariateNormalDiag(
                loc=[0.0]*model.flat_unconstrained_param_event_ndims + self.extra_ndims,
                scale_diag=[1.0]*model.flat_unconstrained_param_event_ndims + self.extra_ndims)
            )
        extra_ndims: `int`
            Number of extra dimensions for augmented normalizing flows.
        posterior_lift_distribution: `callable`
            A callable returning a `tfp.distributions.Distribution`, implementing a
            conditional lift distribution over extra dimensions for augmented normalizing flows.
            (Default: lambda _: tfd.MultivariateNormalDiag(loc=[0.0]*extra_ndims, scale_diag=[1.0]*extra_ndims))
        """

        super(NormalizingFlow, self).__init__(model=model)
        self.flow_bijector = flow_bijector
        self.base_distribution = base_distribution

        if extra_ndims and extra_ndims < 0:
            raise ValueError('`extra_dims` have to be `None` or  `>=0`.')
        else:
            self.extra_ndims = extra_ndims if extra_ndims else 0

        if not self.base_distribution:
            loc = tf.zeros(shape=(self.unconstrained_event_ndims + self.extra_ndims,), dtype=self.dtype)
            scale = tf.ones_like(loc)
            self.base_distribution = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        transformed = tfd.TransformedDistribution(
            distribution=self.base_distribution, bijector=self.flow_bijector
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