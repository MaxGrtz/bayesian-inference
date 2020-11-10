import functools
from typing import Callable, Any, List

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.inference.vi import SurrogatePosterior, MeanFieldADVI
from bayes_vi.models import BayesianModel

tfd = tfp.distributions
tfb = tfp.bijectors


class VI(Inference):

    def __init__(self,
                 model: BayesianModel,
                 dataset: tf.data.Dataset,
                 surrogate_posterior: SurrogatePosterior = MeanFieldADVI(),
                 divergence_fn: Callable[[Any], Any] = tfp.vi.kl_reverse):
        super(VI, self).__init__(model=model, dataset=dataset)
        try:
            self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]
        except ValueError:
            self.features, self.targets = None, list(dataset.batch(dataset.cardinality()).take(1))[0]
        self.model = model(features=self.features, targets=self.targets)
        self.target_log_prob_fn = lambda *args: self.model.joint_distribution.log_prob(
            dict(**dict(zip(self.model.param_names, args)), y=self.targets))
        self.surrogate_posterior_build = surrogate_posterior(self.model, self.model.constraining_bijectors)
        self.loss = functools.partial(tfp.vi.monte_carlo_variational_loss,
                                      discrepancy_fn=divergence_fn,
                                      use_reparameterization=True)

    def fit(self,
            optimizer: tf.optimizers.Optimizer = tf.optimizers.Adam(),
            num_steps: int = 10000,
            sample_size: int = 1,
            convergence_criterion: tfp.optimizer.convergence_criteria.ConvergenceCriterion = None,
            trainable_variables: List[Any] = None,
            seed: Any = None,
            name: str = 'fit_surrogate_posterior'):
        return self.surrogate_posterior_build, tfp.vi.fit_surrogate_posterior(
            self.target_log_prob_fn, self.surrogate_posterior_build, optimizer, num_steps,
            convergence_criterion=convergence_criterion, variational_loss_fn=self.loss,
            sample_size=sample_size, trainable_variables=trainable_variables, seed=seed, name=name
        )
