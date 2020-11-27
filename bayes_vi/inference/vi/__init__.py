import functools
import collections

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils import to_ordered_dict
from bayes_vi.inference import Inference
from bayes_vi.inference.vi.surrogate_posteriors import SurrogatePosterior, MeanFieldADVI

tfd = tfp.distributions
tfb = tfp.bijectors


class VI(Inference):

    def __init__(self, model, dataset, surrogate_posterior, divergence_fn=tfp.vi.kl_reverse):
        super(VI, self).__init__(model=model, dataset=dataset)

        self.surrogate_posterior = surrogate_posterior

        # take num_datapoints as a single batch and extract features and targets
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]

        # condition model on features
        self.model = model(features=self.features)

        # define target_log_prob as the log probability of the model, being the unnormalized log posterior
        self.target_log_prob = functools.partial(self.model.unnormalized_log_posterior, targets=self.targets)
        # self.target_log_prob_fn = lambda **sample: self.target_log_prob(collections.OrderedDict(**sample))
        self.target_log_prob_fn = lambda sample: self.target_log_prob(
            self.surrogate_posterior.reshape_sample(sample)
        )

        self.loss = functools.partial(
            tfp.vi.monte_carlo_variational_loss, discrepancy_fn=divergence_fn, use_reparameterization=True
        )

    def fit(self, optimizer=tf.optimizers.Adam(), num_steps=10000, sample_size=1,
            convergence_criterion=None, trainable_variables=None, seed=None, name='fit_surrogate_posterior'):

        losses = tfp.vi.fit_surrogate_posterior(
            self.target_log_prob_fn, self.surrogate_posterior.posterior_distribution,
            optimizer, num_steps, convergence_criterion=convergence_criterion,
            variational_loss_fn=self.loss, sample_size=sample_size, trainable_variables=trainable_variables,
            seed=seed, name=name
        )

        return self.surrogate_posterior, losses
