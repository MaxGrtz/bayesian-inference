import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp

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
    likelihood: callable
        A callable taking the model parameters, `features` and `targets` (of the dataset)
        and returning a `tfp.distributions.Distribution`.
    constraining_bijectors: `list` of `tfp.bijectors.Bijector`
        A list of diffeomorphisms defined as `tfp.bijectors.Bijector`
        to transform the parameters into unconstrained space R^n.
    features: `tf.Tensor` or `dict[str, tf.Tensor]`
        A single `tf.Tensor` of all features of the dataset of shape (N,m),
        where N is the number of examples in the dataset (or batch) and m is the the number of features.
        Or a mapping from feature names to a `tf.Tensor` of shape (N,1).
    targets: `tf.Tensor`
        A `tf.Tensor` of all target variables of shape (N,r),
        where N is the number of examples in the dataset (or batch) and r is the number of targets.
    distribution: `tfp.distributions.JointDistributionNamedAutoBatched`
        A joint distribution (`tfp.distributions.JointDistributionNamedAutoBatched`)
        of the `priors` and the `likelihood`, defining the Bayesian model.
    """

    def __init__(self,
                 priors,
                 likelihood,
                 constraining_bijectors):
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
        self.priors = priors
        self.param_names = list(priors.keys())
        self.likelihood = likelihood
        self.constraining_bijectors = constraining_bijectors
        self.features = None
        self.targets = None
        self.distribution = None

    def __call__(self,
                 features,
                 targets):
        """Initializes the `Model` with data (`features` and `targets`) and construct the joint `distribution`.

        Parameters
        ----------
        features: `tf.Tensor` or `dict[str, tf.Tensor]`
            A single `tf.Tensor` of all features of the dataset of shape (N,m),
            where N is the number of examples in the dataset (or batch) and m is the the number of features.
            Or a mapping from feature names to a `tf.Tensor` of shape (N,1).
        targets: `tf.Tensor`
            A `tf.Tensor` of all target variables of shape (N,r),
            where N is the number of examples in the dataset (or batch) and r is the number of targets.

        Returns
        -------
        `Model`
            The `Model` instance initialized with `features` and `targets` and the constructed joint `distribution`.

        """
        self.features = features
        self.targets = targets

        # closure of likelihood over the features and targets
        if self.features is not None:
            likelihood = functools.partial(self.likelihood, features=self.features, targets=self.targets)
        else:
            likelihood = functools.partial(self.likelihood, features=None, targets=self.targets)

        # construct the joint distribution defining the Bayesian model
        self.distribution = tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(
                **self.priors,
                y=likelihood,
            )
        )
        return self
