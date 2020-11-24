import collections
import functools
import inspect

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils import make_transform_fn, to_ordered_dict
from bayes_vi.utils.bijectors import CustomBlockwise


tfd = tfp.distributions
tfb = tfp.bijectors


class Model:
    """A probabilistic model in the Bayesian sense.

    A Bayesian `Model` consists of:
        an `collections.OrderedDict` of prior `tfp.distributions.Distribution`,
        a likelihood function (conditional distribution of the data) and
        a list of constraining `tfp.bijectors.Bijector` (can possibly be inferred in later versions).

    Note: There are various additional attributes derived from those fundamental components.

    Attributes
    ----------
    param_names: `list` of `str`
        A list of the ordered parameter names derived from `priors`.
    priors: `collections.OrderedDict[str, tfp.distributions.Distribution]`
        An ordered mapping from parameter names `str` to `tfp.distributions.Distribution`s
        or callables returning a `tfp.distributions.Distribution` (conditional distributions).
    prior_distribution: `tfp.distributions.JointDistributionNamedAutoBatched`
        A joint distribution of the `priors`.
    likelihood: `callable`
        A `callable` taking the model parameters (and `features` of the dataset for regression models)
        and returning a `tfp.distributions.Distribution` of the data. The distribution has to be
        at least 1-dimensional.
    distribution: `tfp.distributions.JointDistributionNamedAutoBatched`
        A joint distribution of the `priors` and the `likelihood`, defining the Bayesian model.
    is_generative_model: `bool`
        A `bool` indicator whether or not the `Model` is a generative model,
        i.e. the likelihood function has no `features` argument.
    posteriors: `collections.OrderedDict[str, tfp.distributions.Distribution]`
        An ordered mapping from parameter names `str` to posterior `tfp.distributions.Distribution`s.
        The distributions are either variational distributions or `tfp.distributions.Empirical` distributions
        derived from samples. Initialized to be equivalent to the priors. Has to be updated via:
        `update_posterior_distribution_by_samples` or `update_posterior_distribution_by_distribution`.
    posterior_distribution: `tfp.distributions.JointDistributionNamedAutoBatched`
        A joint distribution of the `posteriors`, i.e. the `prior_distribution` conditioned on data.
        Initialized to be equivalent to the `prior_distribution`. Has to be updated via:
        `update_posterior_distribution_by_samples` or `update_posterior_distribution_by_distribution`.
    features: `tf.Tensor` or `dict[str, tf.Tensor]`
        A single `tf.Tensor` of all features of the dataset of shape (N,m),
        where N is the number of examples in the dataset (or batch) and m is the the number of features.
        Or a mapping from feature names to a `tf.Tensor` of shape (N,1). Initialized to None.
        The `Model` instance is callable, taking features as an input. This conditions the model on the features
        and updated the attribute `features`.
    constraining_bijectors: `list` of `tfp.bijectors.Bijector`
        A list of diffeomorphisms defined as `tfp.bijectors.Bijector`
        to transform each parameter into unconstrained space R^n. The semantics are chosen,
        such that the inverse transformation of each bijector unconstrains a parameter sample,
        while the forward transformation constrains the parameter sample to the allowed range.
    unconstrained_event_shapes: `list` of `TensorShape`
        The event shape of each prior sample in unconstrained space
        (after applying the corresponding bijector inverse transformation).
    reshaping_bijectors: `list` of `tfp.bijectors.Reshape`
        A list of reshape bijectors, that flatten and reshape parameter samples in unconstrained space.
    reshape_constrain_bijectors: `list` of `tfp.bijectors.Bijector`
        A list of bijectors, chaining the the corresponding `constraining_bijectors` and `reshaping_bijectors`.
        I.e. tfp.bijectors.Chain([constraining_bijector, reshaping_bijector]) for each parameter.
        The inverse transformation thus applies constraining_bijector.inverse to unconstrain
        the parameter sample first and then reshaping_bijector.inverse to flatten it.
        The forward transformation first reshapes the sample and then constrains the sample to the allowed range.
    unconstrain_state: `callable`
        A `callable`, transforming a prior sample (also referred to as state) of the model
        into unconstrained space by applying the inverse transformations of `reshape_constrain_bijectors`
        to the corresponding parameter sample.
    constrain_state: `callable`
        A `callable`, transforming an unconstrained prior sample of the model
        into the originally constrained space by applying the forward transformations of
        `reshape_constrain_bijectors` to the corresponding parameter sample.
    split_bijector: `tfp.bijectors.Split`
        A bijector, whose forward transform splits a `tf.Tensor` into a `list` of `tf.Tensor`,
        and whose inverse transform merges a `list` of `tf.Tensor` into a single `tf.Tensor`.
        This is used to merge the state parts into a single state in unconstrained space.
    """

    def __init__(self, priors, likelihood, constraining_bijectors):
        """Initializes the a `Model` instance.

        Parameters
        ----------
        priors: `collections.OrderedDict[str, tfp.distributions.Distribution]`
            An ordered mapping from parameter names `str` to `tfp.distributions.Distribution`s
            or callables returning a `tfp.distributions.Distribution` (conditional distributions).
        likelihood: `callable`
            A `callable` taking the model parameters (and `features` of the dataset for regression models)
            and returning a `tfp.distributions.Distribution` of the data. The distribution has to be
            at least 1-dimensional.
        constraining_bijectors: `list` of `tfp.bijectors.Bijector`
            A list of diffeomorphisms defined as `tfp.bijectors.Bijector`
            to transform each parameter into unconstrained space R^n. The semantics are chosen,
            such that the inverse transformation of each bijector unconstrains a parameter sample,
            while the forward transformation constrains the parameter sample to the allowed range.
        """
        self.param_names = list(priors.keys())
        self.priors = collections.OrderedDict(priors)
        self.prior_distribution = tfd.JointDistributionNamedAutoBatched(self.priors)
        self.likelihood = likelihood
        self.distribution = tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(
                **self.priors,
                y=likelihood,
            )
        )
        self.is_generative_model = 'features' not in inspect.signature(likelihood).parameters.keys()

        self.posteriors = None
        self.posterior_distribution = None
        self.update_posterior_distribution_by_distribution(self.prior_distribution)

        self.features = None
        self.constraining_bijectors = constraining_bijectors

        prior_sample = list(self.prior_distribution.sample().values())

        # shapes of each sample part in unconstrained sample space
        self.unconstrained_event_shapes = [
            bij.inverse(part).shape
            for part, bij in zip(prior_sample, self.constraining_bijectors)
        ]

        # reshaping bijector from flat to unconstrained sample shape
        self.reshaping_unconstrained_bijectors = [
            tfb.Reshape(event_shape_out=shape, event_shape_in=(-1,))
            for shape in self.unconstrained_event_shapes
        ]

        # reshaping bijector from flat to constrained sample shape
        self.reshaping_constrained_bijectors = [
            tfb.Reshape(event_shape_out=shape, event_shape_in=(-1,))
            for shape in self.prior_distribution.event_shape.values()
        ]

        # reshape flattened unconstrained sample, constrain sample and flatten constrained sample
        self.reshape_constraining_bijectors = [
            tfb.Chain([tfb.Invert(reshape_constrained), constrain, reshape_unconstrained])
            for reshape_constrained, constrain, reshape_unconstrained
            in zip(self.reshaping_constrained_bijectors,
                   self.constraining_bijectors,
                   self.reshaping_unconstrained_bijectors)
        ]

        self.flatten_constrained_sample = make_transform_fn(
            self.reshaping_constrained_bijectors, direction='inverse'
        )
        self.reshape_flat_constrained_sample = make_transform_fn(
            self.reshaping_constrained_bijectors, direction='forward'
        )

        self.flatten_unconstrained_sample = make_transform_fn(
            self.reshaping_unconstrained_bijectors, direction='inverse'
        )
        self.reshape_flat_unconstrained_sample = make_transform_fn(
            self.reshaping_unconstrained_bijectors, direction='forward'
        )

        self.unconstrain_sample = make_transform_fn(
            self.constraining_bijectors, direction='inverse'
        )
        self.constrain_sample = make_transform_fn(
            self.constraining_bijectors, direction='forward'
        )

        self.reshape_unconstrain_sample = make_transform_fn(
            self.reshape_constraining_bijectors, direction='inverse'
        )
        self.reshape_constrain_sample = make_transform_fn(
            self.reshape_constraining_bijectors, direction='forward'
        )

        input_block_sizes = [part.shape[-1]
                             for part in self.flatten_unconstrained_sample(self.unconstrain_sample(prior_sample))]

        output_block_sizes = [part.shape[-1]
                              for part in self.flatten_constrained_sample(prior_sample)]

        self.blockwise_constraining_bijector = CustomBlockwise(
            input_block_sizes=input_block_sizes,
            output_block_sizes=output_block_sizes,
            bijectors=self.reshape_constraining_bijectors
        )

        self.split_unconstrained_bijector = tfb.Split(
            input_block_sizes
        )
        self.split_constrained_bijector = tfb.Split(
            output_block_sizes
        )

    def __call__(self, features):
        """Conditions the `Model` on `features` and updates the joint `distribution`.

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

    def update_posterior_distribution_by_samples(self, posterior_samples):
        """Updates the `posteriors` and the `posterior_distribution` based on `posterior_samples`.

        Parameters
        ----------
        posterior_samples: `list` of `tf.Tensor` or `collections.OrderedDict[str, tf.Tensor]`
            A list or ordered mapping of posterior samples for each parameter.
            E.g. obtained via MCMC or other inference algorithms.
            Providing a single sample for each parameter is also valid
            and corresponds to a point estimate for the parameters.
            The samples are used to construct `tfp.distributions.Empirical` distributions
            for each parameter and the corresponding `tfp.distributions.JointDistributionNamedAutoBatched`.
        """
        if isinstance(posterior_samples, list):
            posterior_samples = to_ordered_dict(self.param_names, posterior_samples)
        if not isinstance(posterior_samples, collections.OrderedDict):
            raise TypeError("`posterior_samples` have to be of type `list` or `collections.OrderedDict`.")

        self.posteriors = collections.OrderedDict(
            [(name, tfd.Empirical(tf.reshape(part, shape=(-1, *list(event_shape))), event_ndims=len(event_shape)))
             for (name, part), event_shape
             in zip(posterior_samples.items(), self.prior_distribution.event_shape.values())]
        )
        self.posterior_distribution = tfd.JointDistributionNamedAutoBatched(self.posteriors)

    def update_posterior_distribution_by_distribution(self, posterior_distribution):
        """Updates the `posteriors` and the `posterior_distribution` based on a `posterior_distribution`.

        TODO: This should allow multivariate distributions in general (not only joint distributions).
              - approx marginal distributions with samples and use them to construct `tfp.distributions.Empirical`
                distributions and in turn a joint distribution ???

        Parameters
        ----------
        posterior_distribution: `tfp.distributions.JointDistributionNamed`
            A joint named distribution (may be auto-batched) e.g. obtained from a variational inference
            algorithm. The joint distribution is used to obtain the component `posteriors`.
        """
        if not isinstance(posterior_distribution, (tfd.JointDistributionNamed, tfd.JointDistributionNamedAutoBatched)):
            raise TypeError("The `posterior_distribution` has to be a `tfp.distributions.JointDistributionNamed` "
                            "or a `tfp.distributions.JointDistributionNamedAutoBatched`.")
        self.posterior_distribution = posterior_distribution

        self.posteriors, _ = self.posterior_distribution.sample_distributions()

    @tf.function
    def unnormalized_log_posterior_parts(self, prior_sample, targets):
        """Computes the unnormalized log posterior parts (prior log prob, data log prob).

        Parameters
        ----------
        prior_sample: `collections.OrderedDict[str, tf.Tensor]`
            A sample from `prior_distribution` with sample_shape=(m,n,...,k).
            That is, `prior_sample` has shape=(m,n,...,k,B,E), where B are the batch
            and E the event dimensions.
        targets: `tf.Tensor`
            A `tf.Tensor` of all target variables of shape (N,r),
            where N is the number of examples in the dataset (or batch) and r is the number of targets.

        Returns
        -------
        `tuple` of `tf.Tensor`
            A tuple consisting of the prior and data log probabilities of the `Model`,
            all of shape (m,n,...,k).
        """
        state = prior_sample.copy()

        event_shape_ndims_first_param = len(list(self.distribution.event_shape.values())[0])

        if event_shape_ndims_first_param > 0:
            sample_shape = list(state.values())[0].shape[:-event_shape_ndims_first_param]
        else:
            sample_shape = list(state.values())[0].shape

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
        prior_sample: `collections.OrderedDict[str, tf.Tensor]`
            A sample from `prior_distribution` with sample_shape=(m,n,...,k).
            That is, `prior_sample` has shape=(m,n,...,k,B,E), where B are the batch
            and E the event dimensions.
        targets: `tf.Tensor`
            A `tf.Tensor` of all target variables of shape (N,r),
            where N is the number of examples in the dataset (or batch) and r is the number of targets.

        Returns
        -------
        `tf.Tensor`
            The unnormalized log posterior probability of the `Model` of shape (m,n,...,k).
        """
        return tf.reduce_sum(list(self.unnormalized_log_posterior_parts(prior_sample, targets)), axis=0)

    @tf.function
    def log_likelihood(self, prior_sample, targets):
        """Computes the log likelihood (data log prob).

        Parameters
        ----------
        prior_sample: `collections.OrderedDict[str, tf.Tensor]`
            A sample from `prior_distribution` with sample_shape=(m,n,...,k).
            That is, `prior_sample` has shape=(m,n,...,k,B,E), where B are the batch
            and E the event dimensions.
        targets: `tf.Tensor`
            A `tf.Tensor` of all target variables of shape (N,r),
            where N is the number of examples in the dataset (or batch) and r is the number of targets.

        Returns
        -------
        `tf.Tensor`
            The log likelihood of the `Model` of shape (m,n,...,k).
        """
        state = prior_sample.copy()

        event_shape_ndims_first_param = len(list(self.distribution.event_shape.values())[0])

        if event_shape_ndims_first_param > 0:
            sample_shape = list(state.values())[0].shape[:-event_shape_ndims_first_param]
        else:
            sample_shape = list(state.values())[0].shape

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

    def sample_prior_predictive(self, shape=()):
        """Generates prior predictive samples.

        Parameters
        ----------
        shape: `tuple`
            Shape of the prior predictive samples to generate.

        Returns
        -------
        `tf.Tensor`
            prior predictive samples of the specified sample shape.
        """
        return self.distribution.sample(shape)['y']

    def sample_posterior_predictive(self, shape=()):
        """Generates posterior predictive samples.

        Parameters
        ----------
        shape: `tuple`
            Shape of the posterior predictive samples to generate.

        Returns
        -------
        `tf.Tensor`
            posterior predictive samples of the specified sample shape.
        """
        if not self.is_generative_model:
            likelihood = functools.partial(self.likelihood, features=self.features)
        else:
            likelihood = self.likelihood

        posterior_model = tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(
                **self.posteriors,
                y=likelihood,
            )
        )
        return posterior_model.sample(shape)['y']

    def transform_sample_forward(self, sample):
        """Convenience function to transform an unconstrained sample into a constrained sample.

        Parameters
        ----------
        sample: `tf.Tensor` or `list` of `tf.Tensor` or `collection.OrderedDict[str, tf.Tensor]`
            A prior sample in unconstrained space, either split into parts or merged.

        Returns
        -------
        `tf.Tensor` or `list` of `tf.Tensor` or `collections.OrderedDict[str, tf.Tensor]`
            The constrained sample, either merged or as a list or ordered mapping of its parts.
        """
        if isinstance(sample, collections.OrderedDict):
            sample_ = list(sample.values())
            to_dict = True
        elif isinstance(sample, (collections.abc.Iterable, tf.Tensor, tf.Variable)):
            sample_ = sample
            to_dict = False
        else:
            raise TypeError("`sample` has to be a tf.Tensor or a collections.OrderedDict or list thereof.")

        if isinstance(sample_, (tf.Tensor, tf.Variable)):
            # for merged unconstrained sample
            sample_ = self.blockwise_constraining_bijector.forward(sample_)
        else:
            try:
                # for flattened unconstrained sample parts
                sample_ = self.reshape_constrain_sample(sample_)
            except:
                # for non flattened unconstrained sample parts
                sample_ = self.constrain_sample(sample_)

        if to_dict:
            return to_ordered_dict(self.param_names, sample_)
        else:
            return sample_

    def target_log_prob_correction_forward(self, sample):
        if isinstance(sample, collections.OrderedDict):
            sample_ = list(sample.values())
        elif isinstance(sample, (collections.abc.Iterable, tf.Tensor, tf.Variable)):
            sample_ = sample
        else:
            raise TypeError("`sample` has to be a tf.Tensor or a collections.OrderedDict or list thereof.")

        if isinstance(sample_, (tf.Tensor, tf.Variable)):
            # for merged unconstrained sample
            return self.blockwise_constraining_bijector.forward_log_det_jacobian(sample_, event_ndims=1)
        else:
            try:
                # for flattened unconstrained sample parts
                return sum(
                    bij.forward_log_det_jacobian(x, event_ndims=1)
                    for bij, x in zip(self.reshape_constraining_bijectors, sample_)
                )
            except:
                # for non flattened unconstrained sample parts
                return sum(
                    bij.forward_log_det_jacobian(x, event_ndims=len(event_shape))
                    for bij, x, event_shape in zip(self.constraining_bijectors, sample_,
                                                   self.unconstrained_event_shapes)
                )

    def transform_sample_inverse(self, sample):
        """Convenience function to transform a constrained sample into an unconstrained sample.

        Note: This is the inverse to `transform_sample_forward`.

        Parameters
        ----------
        sample: `tf.Tensor` or `list` of `tf.Tensor` or `collection.OrderedDict[str, tf.Tensor]`
            A prior sample in constrained space, either split into parts or merged.

        Returns
        -------
        `tf.Tensor` or `list` of `tf.Tensor` or `collections.OrderedDict[str, tf.Tensor]`
            The unconstrained sample, either merged or as a list or ordered mapping of its parts.
        """
        if isinstance(sample, collections.OrderedDict):
            sample_ = list(sample.values())
            to_dict = True
        elif isinstance(sample, (collections.abc.Iterable, tf.Tensor, tf.Variable)):
            sample_ = sample
            to_dict = False
        else:
            raise TypeError("`sample` has to be a tf.Tensor or a collections.OrderedDict or list thereof.")

        if isinstance(sample_, (tf.Tensor, tf.Variable)):
            # for merged constrained sample
            sample_ = self.split_unconstrained_bijector.inverse(
                self.reshape_unconstrain_sample(self.split_constrained_bijector.forward(sample_))
            )
        else:
            try:
                # for flattened constrained sample parts
                sample_ = self.reshape_unconstrain_sample(sample_)
            except:
                # for non flattened constrained sample parts
                sample_ = self.unconstrain_sample(sample_)

        if to_dict:
            return to_ordered_dict(self.param_names, sample_)
        else:
            return sample_

    def target_log_prob_correction_inverse(self, sample):
        if isinstance(sample, collections.OrderedDict):
            sample_ = list(sample.values())
        elif isinstance(sample, (collections.abc.Iterable, tf.Tensor, tf.Variable)):
            sample_ = sample
        else:
            raise TypeError("`sample` has to be a tf.Tensor or a collections.OrderedDict or list thereof.")

        if isinstance(sample_, (tf.Tensor, tf.Variable)):
            # for merged constrained sample
            return self.blockwise_constraining_bijector.inverse_log_det_jacobian(sample_, event_ndims=1)
        else:
            try:
                # for flattened constrained sample parts
                return sum(
                    bij.inverse_log_det_jacobian(x, event_ndims=1)
                    for bij, x in zip(self.reshape_constraining_bijectors, sample_)
                )
            except:
                # for non flattened constrained sample parts
                return sum(
                    bij.inverse_log_det_jacobian(x, event_ndims=len(event_shape))
                    for bij, x, event_shape in zip(self.constraining_bijectors, sample_,
                                                   self.prior_distribution.event_shape.values())
                )
