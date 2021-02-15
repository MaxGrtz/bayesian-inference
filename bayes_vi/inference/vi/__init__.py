import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from fastprogress import fastprogress

from bayes_vi.inference import Inference
from bayes_vi.inference.vi.surrogate_posteriors import SurrogatePosterior

tfd = tfp.distributions
tfb = tfp.bijectors


class VI(Inference):

    def __init__(self, model, dataset, surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse):
        super(VI, self).__init__(model=model, dataset=dataset)

        self.surrogate_posterior = surrogate_posterior

        # take num_datapoints as a single batch and extract features and targets
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]

        # condition model on features
        self.model(features=self.features)

        # define target_log_prob as the log probability of the model, being the unnormalized log posterior
        self.target_log_prob = functools.partial(self.model.unnormalized_log_posterior, targets=self.targets)

        self.target_log_prob_fn = self.surrogate_posterior.get_target_log_prob_fn(self.target_log_prob)

        self.discrepancy_fn = discrepancy_fn

    def safe_discrepancy_fn(self, logu):
        discrepancy = self.discrepancy_fn(logu)
        return tf.boolean_mask(discrepancy, tf.math.is_finite(discrepancy))

    @staticmethod
    def make_optimizer_step_fn(loss_fn, optimizer):

        @tf.function(autograph=False)
        def optimizer_step():
            with tf.GradientTape() as tape:
                loss = loss_fn()

            grads = tape.gradient(loss, tape.watched_variables())
            clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)

            optimizer.apply_gradients(zip(clipped_grads, tape.watched_variables()))
            return loss

        return optimizer_step

    def fit(self, optimizer=tf.optimizers.Adam(), num_steps=10000, sample_size=1,
            num_samples_to_approx_marginals=10000, progress_bar=True):

        losses = np.zeros(num_steps)

        def loss_fn():
            return tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn=self.target_log_prob_fn,
                surrogate_posterior=self.surrogate_posterior.posterior_distribution,
                discrepancy_fn=self.safe_discrepancy_fn,
                use_reparameterization=True,
                sample_size=sample_size
            )

        if progress_bar:
            steps = fastprogress.progress_bar(range(num_steps))
        else:
            steps = range(num_steps)

        optimizer_step = self.make_optimizer_step_fn(loss_fn, optimizer)

        for i in steps:
            loss = optimizer_step()
            losses[i] = loss
            if i % 10 == 0 and hasattr(steps, "comment"):
                steps.comment = "avg loss: {:.3f}".format(losses[max(0, i - 100):i + 1].mean())

        approx_posterior = self.surrogate_posterior.approx_joint_marginal_posteriors(num_samples_to_approx_marginals)
        return approx_posterior, losses
