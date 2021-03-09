"""Module providing the Model class defining a Bayesian statistical model."""

import collections
import functools
import inspect

import decorator
import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils import to_ordered_dict

tfd = tfp.distributions
tfb = tfp.bijectors


@decorator.decorator
def sample(f, num_samples=1, *args, **kwargs):
    """Decorator to map data distribution onto the product data distribution based on num_samples."""
    llh = f(*args, **kwargs)
    return tfd.Sample(llh, sample_shape=num_samples, name='Sample_Likelihood')


class Model:
    """A probabilistic model in the Bayesian sense.

    A Bayesian `Model` consists of:
        an `collections.OrderedDict` of prior `tfp.distributions.Distribution`,
        a likelihood function (conditional distribution of the data) and
        a list of constraining `tfp.bijectors.Bijector`.

    Note: There are various additional attributes derived from those fundamental components.

    Attributes
    ----------
    likelihood: `callable`
        A `callable` taking the model parameters (and `features` of the dataset for regression model)
        and returning a `tfp.distributions.Distribution` of the data.
    is_generative_model: `bool`
        A `bool` indicator whether or not the `Model` is a generative model,
        i.e. the likelihood function has no `features` argument.
    priors: `collections.OrderedDict[str, tfp.distributions.Distribution]`
        An ordered mapping from parameter names `str` to `tfp.distributions.Distribution`s
        or callables returning a `tfp.distributions.Distribution` (conditional distributions).
    param_names: `list` of `str`
        A list of the ordered parameter names derived from `priors`.
    dtypes: `list` of `tf.dtype`
        A list of the dtype of each parameter.
    prior_distribution: `tfp.distributions.JointDistributionNamedAutoBatched`
        A joint distribution of the `priors`.
    constraining_bijectors: `list` of `tfp.bijectors.Bijector`
        A list of diffeomorphisms defined as `tfp.bijectors.Bijector`
        to transform each parameter into unconstrained space R^n. The semantics are chosen,
        such that the inverse transformation of each bijector unconstrains a parameter sample,
        while the forward transformation constrains the parameter sample to the allowed range.
    joint_constraining_bijector: `tfp.bijectors.JointMap`
        A bijector which applies the constraining bijectors to the respective parameters in parallel.
    param_event_shape: `list` of `tf.TensorShape`
        A list of the event shapes of each prior distribution.
    unconstrained_param_event_shape: `list` of `tf.TensorShape`
        A list of the prior distributions event shapes in unconstrained space.
    reshape_flat_param_bijector: `tfp.bijectors.JointMap`
        A bijector which reshapes (forward) and flattens (inverse) a parameter samples in constrained space.
    reshape_flat_unconstrained_param_bijector: `tfp.bijectors.JointMap`
        A bijector which reshapes (forward) and flattens (inverse) a parameter samples in unconstrained space.
    flat_param_event_shape: `list` of `tf.TensorShape`
        A list of flattened parameter event shapes in constrained space.
    flat_unconstrained_param_event_shape: `list` of `tf.TensorShape`
        A list of flattened parameter event shapes in unconstrained space.
    split_flat_param_bijector: `tfp.bijectors.Split`
        A bijector which concatenates (inverse) a the parts of a parameter sample and
        splits (forward) such a merged sample in constrained space.
    split_flat_unconstrained_param_bijector: `tfp.bijectors.Split`
        A bijector which concatenates (inverse) a the parts of a parameter sample and
        splits (forward) such a merged sample in unconstrained space.
    blockwise_constraining_bijector: `tfp.bijectors.Chain`
        A chain of bijectors, that splits, reshapes, constrains, flattens and concatenates
        a merged sample on the forward transformation.
    flat_param_event_ndims: `int`
        The dimension of the constrained parameter space.
    flat_unconstrained_param_event_ndims: `int`
        The dimension of the unconstrained parameter space.
    """

    def __init__(self, priors, likelihood, constraining_bijectors=None):
        """Initializes the a `Model` instance.

        Parameters
        ----------
        priors: `collections.OrderedDict[str, tfp.distributions.Distribution]`
            An ordered mapping from parameter names `str` to `tfp.distributions.Distribution`s
            or callables returning a `tfp.distributions.Distribution` (conditional distributions).
        likelihood: `callable`
            A `callable` taking the model parameters (and `features` of the dataset for regression model)
            and returning a `tfp.distributions.Distribution` of the data.
        constraining_bijectors: `list` of `tfp.bijectors.Bijector`
            A list of diffeomorphisms defined as `tfp.bijectors.Bijector`
            to transform each parameter into unconstrained space R^n. The semantics are chosen,
            such that the inverse transformation of each bijector unconstrains a parameter sample,
            while the forward transformation constrains the parameter sample to the allowed range.
            If no list is provided, default bijectors are used.
        """
        self.likelihood = likelihood
        self.is_generative_model = 'features' not in inspect.signature(likelihood).parameters.keys()

        self.param_names = list(priors.keys())
        self.priors = collections.OrderedDict(priors)
        self.dtypes = collections.OrderedDict([(k, v.dtype) for k, v in priors.items()])
        self.prior_distribution = tfd.JointDistributionNamedAutoBatched(self.priors)
        self.constraining_bijectors = constraining_bijectors

        if not self.constraining_bijectors:
            self.constraining_bijectors = list(
                self.prior_distribution.experimental_default_event_space_bijector().bijectors
            )
        self.joint_constraining_bijector = tfb.JointMap(self.constraining_bijectors)

        self.param_event_shape = list(self.prior_distribution.event_shape.values())
        self.unconstrained_param_event_shape = self.joint_constraining_bijector.inverse_event_shape(
            self.param_event_shape
        )

        self.reshape_flat_param_bijector = tfb.JointMap([
            tfb.Reshape(event_shape_out=shape, event_shape_in=(-1,)) for shape in self.param_event_shape
        ])
        self.reshape_flat_unconstrained_param_bijector = tfb.JointMap([
            tfb.Reshape(event_shape_out=shape, event_shape_in=(-1,)) for shape in self.unconstrained_param_event_shape
        ])

        prior_sample = list(self.prior_distribution.sample().values())

        self.flat_param_event_shape = [
            part.shape for part in self.reshape_flat_param_bijector.inverse(
                prior_sample
            )
        ]

        self.flat_unconstrained_param_event_shape = [
            part.shape for part in self.reshape_flat_unconstrained_param_bijector.inverse(
                self.joint_constraining_bijector.inverse(prior_sample)
            )
        ]

        block_sizes = [shape[-1] for shape in self.flat_param_event_shape]
        unconstrained_block_sizes = [shape[-1] for shape in self.flat_unconstrained_param_event_shape]

        self.split_flat_param_bijector = tfb.Split(
            block_sizes
        )

        self.split_flat_unconstrained_param_bijector = tfb.Split(
            unconstrained_block_sizes
        )

        self.blockwise_constraining_bijector = tfb.Chain([
            tfb.Invert(self.split_flat_param_bijector),
            tfb.Invert(self.reshape_flat_param_bijector),
            self.joint_constraining_bijector,
            self.reshape_flat_unconstrained_param_bijector,
            self.split_flat_unconstrained_param_bijector
        ])

        self.flat_param_event_ndims = sum(
            block_sizes
        )

        self.flat_unconstrained_param_event_ndims = sum(
            unconstrained_block_sizes
        )

    def _get_joint_distribution(self, param_distributions, num_samples=None, features=None):
        """Utility function returning the joint distribution of the model.

        This function is primarily for internal use.
        It handles the differences between generative and regression model.
        If generative model -> provide `num_samples` over which to construct the product joint distribution.
        If regression model -> provide `features` to condition the model on.

        Parameters
        ----------
        param_distributions: `collections.OrderedDict[str, tfp.distributions.Distribution]`
            An ordered mapping from parameter names `str` to `tfp.distributions.Distribution`s
            or callables returning a `tfp.distributions.Distribution` (conditional distributions).
        num_samples: `int`
            The number of samples over which to construct the product joint distribution.
        features: `tf.Tensor` or `dict[str, tf.Tensor]`
            The features on which to condition the model.

        Returns
        -------
        `tfp.distributions.JointDistributionNamedAutoBatched`
            The respective joint distribution of the model conditioned on `features` if provided.
        """
        if not self.is_generative_model:
            llh = functools.partial(self.likelihood, features=features)
        else:
            if num_samples is not None:
                llh = sample(self.likelihood, num_samples=num_samples)
            else:
                llh = self.likelihood
        return tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(**param_distributions, y=llh)
        )

    def get_param_distributions(self, joint_param_distribution=None, param_samples=None):
        """Utility function returning the marginal parameter distributions given a joint distribution or samples.

        This function is primarily for internal use.

        Parameters
        ----------
        joint_param_distribution: `tfp.distributions.JointDistributionNamed` or `tfp.distributions.JointDistributionNamedAutoBatched`
            A joint distribution over the model parameters (e.g. prior or posterior)
        param_samples: `list` of `tf.Tensor` or `collections.OrderedDict[str, tf.Tensor]`
            A list or collections.OrderedDict of parameter samples.

        Returns
        -------
        `collections.OrderedDict[str, tfp.distributions.Distribution]`
            A mapping from parameter names to their marginal distributions.
        """
        if isinstance(joint_param_distribution, (tfd.JointDistributionNamed, tfd.JointDistributionNamedAutoBatched)):
            param_dists, _ = joint_param_distribution.sample_distributions()
        elif isinstance(param_samples, (list, collections.OrderedDict)):
            if isinstance(param_samples, list):
                param_samples = to_ordered_dict(self.param_names, param_samples)
            param_dists = collections.OrderedDict(
                [(name, tfd.Empirical(tf.reshape(part, shape=(-1, *list(event_shape))), event_ndims=len(event_shape)))
                 for (name, part), event_shape
                 in zip(param_samples.items(), self.param_event_shape)]
            )
        else:
            raise ValueError('You have to provide either a joint distribution or param samples.')
        return param_dists

    def get_joint_distribution(self, num_samples=None, features=None):
        """Constructs the joint distribution of the model.

        Parameters
        ----------
        num_samples: `int`
            The number of samples over which to construct the product joint distribution.
        features: `tf.Tensor` or `dict[str, tf.Tensor]`
            The features on which to condition the model.

        Returns
        -------
        `tfp.distributions.JointDistributionNamedAutoBatched`
            The respective joint distribution of the model conditioned on `features` if provided.
        """
        return self._get_joint_distribution(self.priors, num_samples=num_samples, features=features)

    def get_posterior_predictive_distribution(self, posterior_distribution=None, posterior_samples=None,
                                              num_samples=None, features=None):
        """Constructs the posterior predictive distribution of the model.

        Parameters
        ----------
        posterior_distribution: `tfp.distributions.JointDistributionNamed` or `tfp.distributions.JointDistributionNamedAutoBatched`
            A joint distribution over the model parameters (e.g. prior or posterior)
        posterior_samples: `list` of `tf.Tensor` or `collections.OrderedDict[str, tf.Tensor]`
            A list or collections.OrderedDict of parameter samples.
        num_samples: `int`
            The number of samples over which to construct the product joint distribution.
        features: `tf.Tensor` or `dict[str, tf.Tensor]`
            The features on which to condition the model.

        Returns
        -------
        `tfp.distributions.JointDistributionNamedAutoBatched`
            The posterior predictive distribution of the model conditioned on `features` if provided.
        """
        posteriors = self.get_param_distributions(
            joint_param_distribution=posterior_distribution,
            param_samples=posterior_samples
        )
        return self._get_joint_distribution(posteriors, num_samples=num_samples, features=features)
