import functools
import tqdm

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.metrics import Mean


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


    def fit(self, optimizer=tf.optimizers.Adam(), num_steps=10000, sample_size=1, trace_fn=lambda trace: trace.loss,
            convergence_criterion=None, trainable_variables=None, seed=None, name='fit_surrogate_posterior'):
        loss_fn = functools.partial(
            tfp.vi.monte_carlo_variational_loss, discrepancy_fn=self.discrepancy_fn, use_reparameterization=True
        )
        trace = self._fit(self.target_log_prob_fn, self.surrogate_posterior.posterior_distribution,
                          optimizer, num_steps, convergence_criterion, loss_fn, sample_size, trace_fn,
                          trainable_variables, seed, name)
        return self.surrogate_posterior, trace


    @staticmethod
    @tf.function
    def _fit(target_log_prob_fn, surrogate_posterior, optimizer, num_steps, convergence_criterion,
             loss, sample_size, trace_fn, trainable_variables, seed, name):
        return tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn, surrogate_posterior, optimizer, num_steps, sample_size=sample_size,
            convergence_criterion=convergence_criterion, variational_loss_fn=loss, trace_fn=trace_fn,
            trainable_variables=trainable_variables, seed=seed, name=name
        )



    def custom_fit(self, optimizer=tf.optimizers.Adam(), num_steps=10000, sample_size=1, mle_fit=False):
        losses = []
        loss_mean = Mean()

        if mle_fit:
            loss_fn = lambda q_samples: - tf.reduce_mean(
                self.target_log_prob_fn(q_samples)
            )
        else:
            divergence_fn = lambda q_samples: self.discrepancy_fn(
                    self.target_log_prob_fn(q_samples) - self.surrogate_posterior.posterior_distribution.log_prob(q_samples)
                )
            loss_fn = lambda q_samples: tfp.monte_carlo.expectation(
                f=divergence_fn,
                samples=q_samples,
                use_reparameterization=True)

        t = tqdm.trange(num_steps)
        for _ in t:
            t.set_postfix(loss=loss_mean.result().numpy())
            with tf.GradientTape() as tape:
                samples = self.surrogate_posterior.posterior_distribution.sample(sample_size)
                loss = loss_fn(samples)
            losses.append(loss)
            loss_mean(loss)

            # compute gradients
            grads = tape.gradient(loss, tape.watched_variables())
            clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]

            # adjust variables by applying gradient descent update
            optimizer.apply_gradients(zip(clipped_grads, tape.watched_variables()))
        return self.surrogate_posterior, losses
