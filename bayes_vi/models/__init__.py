import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils import make_transform_fn, to_ordered_dict

import inspect


tfd = tfp.distributions
tfb = tfp.bijectors


class Model:
    """A probabilistic model in the Bayesian sense.

    A Bayesian `Model` consists of:
        a likelihood function (conditional distribution of the data),
        an `collections.OrderedDict` of prior `tfp.distributions.Distribution` and
        a list of constraining `tfp.bijectors.Bijector` (can possibly be inferred in later versions).

    Attributes
    ----------
    priors: `collections.OrderedDict[str, tfp.distributions.Distribution]`
        An ordered mapping from parameter names `str` to `tfp.distributions.Distribution`s
        or callables returning a `tfp.distributions.Distribution` (conditional distributions).
        The `tfp.distributions.Distribution`s may contain trainable hyperparameters
        as `tf.Variable` or `tfp.util.TransformedVariable`.
    param_names: `list` of `str`
        A list of the ordered parameter names derived from `priors`.
    prior_distribution: `tfp.distributions.JointDistributionNamedAutoBatched`
        A joint distribution of the `priors`.
    constraining_bijectors: `list` of `tfp.bijectors.Bijector`
        A list of diffeomorphisms defined as `tfp.bijectors.Bijector`
        to transform the parameters into unconstrained space R^n.
    likelihood: `callable`
        A `callable` taking the model parameters, `features` and `targets` (of the dataset)
        and returning a `tfp.distributions.Distribution`.
    is_generative_model: `bool`
        A `bool` indicator whether or not the `Model` is a generative model,
        i.e. the likelihood function has no `features` argument.
    features: `tf.Tensor` or `dict[str, tf.Tensor]`
        A single `tf.Tensor` of all features of the dataset of shape (N,m),
        where N is the number of examples in the dataset (or batch) and m is the the number of features.
        Or a mapping from feature names to a `tf.Tensor` of shape (N,1).
    distribution: `tfp.distributions.JointDistributionNamedAutoBatched`
        A joint distribution of the `priors` and the `likelihood`, defining the Bayesian model.
    unconstrain_state: `callable`
        A `callable`, transforming a prior sample of `Model` into unconstrained space.
    constrain_state: `callable`
        A `callable`, transforming an unconstrained prior sample
        of `Model` into the originally constrained space.
    split_bijector: `tfp.bijectors.Split`
        A bijector, whose forward transform splits a `tf.Tensor` into a `list` of `tf.Tensor`,
        and whose inverse transform merges a `list` of `tf.Tensor` into a single `tf.Tensor`.
        Note: this is used to merge the state parts into a single state.
    """

    def __init__(self, priors, likelihood, constraining_bijectors):
        """Initializes the a `Model` instance.

        Parameters
        ----------
        priors: `collections.OrderedDict[str, tfp.distributions.Distribution]`
            An ordered mapping from parameter names (`str`) to `tfp.distributions.Distribution`
            or callables returning a `tfp.distributions.Distribution` (conditional distributions).
        likelihood: callable
            A callable taking the model parameters, features and targets (of the dataset)
            and returning a `tfp.distributions.Distribution`.
        constraining_bijectors: `list` of `tfp.bijectors.Bijector`
            A list of diffeomorphisms defined as `tfp.bijectors.Bijector`
            to transform the parameters into unconstrained space R^n.
        """
        self.priors = collections.OrderedDict([(k, tfd.Sample(v, sample_shape=1))
                                               if not callable(v) and v.event_shape == [] and v.batch_shape == []
                                               else (k, v)
                                               for k, v in priors.items()])
        self.param_names = list(self.priors.keys())
        self.prior_distribution = tfd.JointDistributionNamedAutoBatched(self.priors)
        self.posteriors = None
        self.posterior_distribution = None
        self.posterior_model = None
        self.constraining_bijectors = constraining_bijectors
        self.likelihood = likelihood
        self.is_generative_model = 'features' not in inspect.signature(likelihood).parameters.keys()
        self.features = None
        self.distribution = tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(
                **self.priors,
                y=likelihood,
            )
        )
        self.init_posterior_model_by_distribution(self.prior_distribution)

        prior_sample = list(self.prior_distribution.sample().values())

        self.unconstrained_event_shapes = [
            bij.inverse(part).shape
            for part, bij in zip(prior_sample, self.constraining_bijectors)
        ]

        self.reshaping_bijectors = [
            tfb.Reshape(event_shape_out=shape, event_shape_in=(-1,))
            for shape in self.unconstrained_event_shapes
        ]

        self.reshape_constrain_bijectors = [
            tfb.Chain([constr, reshape]) for reshape, constr
            in zip(self.reshaping_bijectors, self.constraining_bijectors)
        ]

        self.unconstrain_state = make_transform_fn(self.reshape_constrain_bijectors, direction='inverse')
        self.constrain_state = make_transform_fn(self.reshape_constrain_bijectors, direction='forward')

        self.split_bijector = tfb.Split(
            [part.shape[-1] for part in self.unconstrain_state(prior_sample)]
        )

    def __call__(self, features):
        """Conditions the `Model` on `features` and update the joint `distribution`.

        Parameters
        ----------
        features: `tf.Tensor` or `dict[str, tf.Tensor]`
            A single `tf.Tensor` of all features of the dataset of shape (N,m),
            where N is the number of examples in the dataset (or batch) and m is the the number of features.
            Or a mapping from feature names to a `tf.Tensor` of shape (N,1).

        Returns
        -------
        `Model`
            The `Model` instance conditioned on `features`.

        """
        if not self.is_generative_model:
            self.features = features

            likelihood = functools.partial(self.likelihood, features=self.features)

            # update the joint distribution defining the Bayesian model
            self.distribution = tfd.JointDistributionNamedAutoBatched(
                collections.OrderedDict(
                    **self.priors,
                    y=likelihood,
                )
            )
        return self

    def init_posterior_model_by_samples(self, posterior_samples):
        self.posteriors = collections.OrderedDict(
            [(name, tfd.Empirical(tf.reshape(part, shape=(-1, *list(event_shape))), event_ndims=len(event_shape)))
             for (name, part), event_shape
             in zip(posterior_samples.items(), self.prior_distribution.event_shape.values())]
        )
        self.posterior_distribution = tfd.JointDistributionNamedAutoBatched(self.posteriors)

        if not self.is_generative_model:
            likelihood = functools.partial(self.likelihood, features=self.features)
        else:
            likelihood = self.likelihood

        self.posterior_model = tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(
                **self.posteriors,
                y=likelihood,
            )
        )

    def init_posterior_model_by_distribution(self, posterior_distribution):
        if not isinstance(posterior_distribution, (tfd.JointDistributionNamed, tfd.JointDistributionNamedAutoBatched)):
            raise TypeError("The `posterior_distribution` has to be a `tfp.distributions.JointDistributionNamed` "
                            "or a `tfp.distributions.JointDistributionNamedAutoBatched`.")
        self.posterior_distribution = posterior_distribution

        self.posteriors, _ = self.posterior_distribution.sample_distributions()

        if not self.is_generative_model:
            likelihood = functools.partial(self.likelihood, features=self.features)
        else:
            likelihood = self.likelihood

        self.posterior_model = tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(
                **self.posteriors,
                y=likelihood,
            )
        )

    @tf.function
    def unnormalized_log_posterior_parts(self, prior_sample, targets):
        """Computes the unnormalized log posterior parts (prior log prob, data log prob).

        Parameters
        ----------
        prior_sample:
            A sample from `prior_distribution` of the `Model`
        targets: `tf.Tensor`
            A `tf.Tensor` of all target variables of shape (N,r),
            where N is the number of examples in the dataset (or batch) and r is the number of targets.

        Returns
        -------
        `tuple` of `tf.Tensor`
            A tuple consisting of the prior and data log probabilities of the `Model`.
        """
        state = prior_sample.copy()

        sample_shape = list(state.values())[0].shape[:-len(list(self.distribution.event_shape.values())[0])]

        if self.features is None:
            state.update(
                y=tf.reshape(targets, shape=[targets.shape[0]] + [1] * len(sample_shape) + targets.shape[1:])
            )
            log_prob_data = tf.reduce_sum(self.distribution.log_prob_parts(state)['y'], axis=0)
            return self.prior_distribution.log_prob(prior_sample), tf.reshape(log_prob_data, shape=sample_shape)
        else:
            state.update(
                y=tf.reshape(targets, shape=[1] * len(sample_shape) + targets.shape)
            )
            log_prob_data = self.distribution.log_prob_parts(state)['y']
            return self.prior_distribution.log_prob(prior_sample), tf.reshape(log_prob_data, shape=sample_shape)

    @tf.function
    def unnormalized_log_posterior(self, prior_sample, targets):
        """Computes the unnormalized log posterior.

        Note: this just sums the results of `unnormalized_log_posterior_parts` (prior log prob + data log prob)

        Parameters
        ----------
        prior_sample:
            A sample from `prior_distribution` of the `Model`
        targets: `tf.Tensor`
            A `tf.Tensor` of all target variables of shape (N,r),
            where N is the number of examples in the dataset (or batch) and r is the number of targets.

        Returns
        -------
        `tf.Tensor`
            The unnormalized log posterior probability of the `Model`.
        """
        return tf.reduce_sum(list(self.unnormalized_log_posterior_parts(prior_sample, targets)), axis=0)

    @tf.function
    def log_likelihood(self, prior_sample, targets):
        """Computes the log likelihood (data log prob).

        Parameters
        ----------
        prior_sample:
            A sample from `prior_distribution` of the `Model`
        targets: `tf.Tensor`
            A `tf.Tensor` of all target variables of shape (N,r),
            where N is the number of examples in the dataset (or batch) and r is the number of targets.

        Returns
        -------
        `tf.Tensor`
            The log likelihood of the `Model`.
        """
        state = prior_sample.copy()

        sample_shape = list(state.values())[0].shape[:-len(list(self.distribution.event_shape.values())[0])]

        if self.features is None:
            state.update(
                y=tf.reshape(targets, shape=[targets.shape[0]] + [1] * len(sample_shape) + targets.shape[1:])
            )
            return tf.reshape(tf.reduce_sum(self.distribution.log_prob_parts(state)['y'], axis=0), shape=sample_shape)
        else:
            state.update(
                y=tf.reshape(targets, shape=[1] * len(sample_shape) + targets.shape)
            )
            return tf.reshape(self.distribution.log_prob_parts(state)['y'], shape=sample_shape)

    @tf.function
    def sample_prior_predictive(self, shape):
        return self.distribution.sample(shape)['y']

    @tf.function
    def sample_posterior_predictive(self, shape):
        return self.posterior_model.sample(shape)['y']

    @tf.function
    def transform_state_forward(self, state, split=True, to_dict=True):
        """Transforms an unconstrained state into a `collections.orderedDict` of constrained state parts."""
        if split:
            _state = self.constrain_state(self.split_bijector.forward(state))
        else:
            _state = self.constrain_state(state)

        if to_dict:
            return to_ordered_dict(self.param_names, _state)
        else:
            return _state

    @tf.function
    def transform_state_inverse(self, state, split=True, from_dict=True):
        """Transforms a `collections.orderedDict` of constrained state parts into an unconstrained state."""
        if from_dict:
            _state = list(state.values())
        else:
            _state = state
        if split:
            return self.split_bijector.inverse(self.unconstrain_state(_state))
        else:
            return self.unconstrain_state(_state)
