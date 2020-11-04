import collections
import functools
from typing import Callable, List, Any, Dict, Union, OrderedDict

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class Model:

    def __init__(self,
                 param_names: List[str],
                 constraining_bijectors: Union[List[tfb.Bijector], None]) -> None:
        self.param_names = param_names
        self.constraining_bijectors = constraining_bijectors

    def __call__(self, *args, **kwargs):
        pass


class FrequentistModel(Model):

    def __init__(self,
                 params: OrderedDict[str, tf.Tensor],
                 likelihood: Callable[[Any], tfd.Distribution],
                 constraining_bijectors: List[tfb.Bijector]) -> None:
        super(FrequentistModel, self).__init__(param_names=list(params.keys()),
                                               constraining_bijectors=constraining_bijectors)
        self.params = collections.OrderedDict(
            [(k, tfp.util.TransformedVariable(v, bijector=bij))
             for (k, v), bij in zip(params.items(), constraining_bijectors)]
        )
        self.likelihood = likelihood
        self.features = None
        self.targets = None
        self.data_distribution = None

    def __call__(self,
                 features: Union[Dict[str, tf.Tensor], tf.Tensor],
                 targets: tf.Tensor) -> Model:
        self.features = features
        self.targets = targets

        if self.features is not None:
            likelihood = functools.partial(self.likelihood, features=self.features, targets=self.targets)
        else:
            likelihood = functools.partial(self.likelihood, features=None, targets=self.targets)

        self.data_distribution = likelihood(**self.params)
        return self


class BayesianModel(Model):

    def __init__(self,
                 priors: OrderedDict[str, Union[tfd.Distribution, Callable[[Any], tfd.Distribution]]],
                 likelihood: Callable[[Any], tfd.Distribution],
                 constraining_bijectors: List[tfb.Bijector]) -> None:
        super(BayesianModel, self).__init__(param_names=list(priors.keys()),
                                            constraining_bijectors=constraining_bijectors)
        self.priors = priors
        self.likelihood = likelihood
        self.features = None
        self.targets = None
        self.joint_distribution = None

    def __call__(self,
                 features: Union[Dict[str, tf.Tensor], tf.Tensor],
                 targets: tf.Tensor) -> Model:
        self.features = features
        self.targets = targets

        if self.features is not None:
            likelihood = functools.partial(self.likelihood, features=self.features, targets=self.targets)
        else:
            likelihood = functools.partial(self.likelihood, features=None, targets=self.targets)

        self.joint_distribution = tfd.JointDistributionNamedAutoBatched(
            collections.OrderedDict(
                **self.priors,
                y=likelihood,
            )
        )
        return self
