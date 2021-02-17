import collections
import functools
import inspect
import decorator

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils import make_transform_fn, to_ordered_dict
from bayes_vi.utils.bijectors import CustomBlockwise

tfd = tfp.distributions
tfb = tfp.bijectors


@decorator.decorator
def sample(f, sample_shape=(), *args, **kwargs):
    llh = f(*args, **kwargs)
    return tfd.Sample(llh, sample_shape=sample_shape, name='Sample_Likelihood')


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
        The event shape of each parameter sample in unconstrained space
        (after applying the corresponding bijector inverse transformation).
    reshaping_unconstrained_bijectors: `list` of `tfp.bijectors.Reshape`
        A list of reshape bijectors, that flatten and reshape parameter samples in unconstrained space.
    reshaping_constrained_bijectors: `list` of `tfp.bijectors.Reshape`
        A list of reshape bijectors, that flatten and reshape parameter samples in constrained space.
    reshape_constraining_bijectors: `list` of `tfp.bijectors.Bijector`
        A list of bijectors, chaining the corresponding `constraining_bijectors` and reshaping bijectors.
        I.e. tfp.bijectors.Chain([tfb.Invert(reshaping_constrained_bij), constrain, reshaping_unconstrained_bij])
        for each parameter. The inverse transformation thus reshapes a flattened sample in constrained space,
        unconstrains it and then flattens it in unconstrained space. The forward transformation first reshapes
        the sample in unconstrained space, constrains it and then flattens it in constrained space.
    flatten_constrained_sample: `callable`
        A `callable`, flattening a constrained parameter sample of the model by applying
        the inverse transformations of `reshaping_constrained_bijectors` to the corresponding
        parameter sample parts.
    flatten_unconstrained_sample: `callable`
        A `callable`, flattening an unconstrained parameter sample of the model by applying
        the inverse transformations of `reshaping_unconstrained_bijectors` to the corresponding
        parameter sample parts.
    reshape_flat_constrained_sample:
        A `callable`, reshaping a flattened constrained parameter sample of the model by applying
        the forward transformations of `reshaping_constrained_bijectors` to the corresponding
        parameter sample parts.
    reshape_flat_unconstrained_sample:
        A `callable`, reshaping a flattened unconstrained parameter sample of the model by applying
        the forward transformations of `reshaping_unconstrained_bijectors` to the corresponding
        parameter sample parts.
    constrain_sample: `callable`
        A `callable`, transforming an unconstrained parameter sample of the model
        into constrained space by applying the forward transformations of
        `constraining_bijectors` to the corresponding parameter sample parts.
    unconstrain_sample: `callable`
        A `callable`, transforming a constrained parameter sample of the model
        into unconstrained space by applying the inverse transformations of
        `constraining_bijectors` to the corresponding parameter sample parts.
    reshape_constrain_sample: `callable`
        A `callable`, transforming a flattened unconstrained parameter sample of the model
        into constrained space by applying the forward transformations of
        `reshape_constraining_bijectors` to the corresponding parameter sample parts.
    reshape_unconstrain_sample: `callable`
        A `callable`, transforming a flattened constrained parameter sample of the model
        into unconstrained space by applying the inverse transformations of
        `reshape_constraining_bijectors` to the corresponding parameter sample parts.
    split_unconstrained_bijector: `tfp.bijectors.Split`
        A bijector, whose forward transform splits a `tf.Tensor` into a `list` of `tf.Tensor`,
        and whose inverse transform merges a `list` of `tf.Tensor` into a single `tf.Tensor`.
        This is used to merge flattened sample parts into a single merged sample in unconstrained space.
    split_constrained_bijector: `tfp.bijectors.Split`
        A bijector, whose forward transform splits a `tf.Tensor` into a `list` of `tf.Tensor`,
        and whose inverse transform merges a `list` of `tf.Tensor` into a single `tf.Tensor`.
        This is used to merge flattened sample parts into a single merged sample in constrained space.
    blockwise_constraining_bijector: `bayes_vi.utils.bijectors.CustomBlockwise`
        A modification of `tfp.bijectors.Blockwise`. This bijectors allows constraining/unconstraining
        a merged parameter sample. Here, a merged parameter sample corresponds to:
            in constrained space:   split_constrained_bijector.inverse(flatten_constrained_sample(sample))
            in unconstrained space: split_unconstrained_bijector.inverse(flatten_unconstrained_sample(sample))
    """

    def __init__(self, priors, likelihood, constraining_bijectors=None):
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


    def _get_joint_distribution(self, param_distributions, targets=None, features=None):
        if not self.is_generative_model:
            llh = functools.partial(self.likelihood, features=features)
        else:
            if targets is not None:
                llh = sample(self.likelihood, sample_shape=targets.shape[0])
            else:
                llh = self.likelihood
        return tfd.JointDistributionNamedAutoBatched(
                collections.OrderedDict(**param_distributions, y=llh)
            )

    def get_param_distributions(self, joint_param_distribution=None, param_samples=None):
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

    def get_joint_distribution(self, targets=None, features=None):
        return self._get_joint_distribution(self.priors, targets=targets, features=features)


    def get_posterior_predictive_distribution(self, posterior_distribution=None, posterior_samples=None, targets=None, features=None):
        posteriors = self.get_param_distributions(
            joint_param_distribution=posterior_distribution,
            param_samples=posterior_samples
        )
        return self._get_joint_distribution(posteriors, targets=targets, features=features)