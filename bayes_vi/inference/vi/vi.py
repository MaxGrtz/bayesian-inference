"""Module providing the VI class."""

import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from fastprogress import fastprogress

from bayes_vi.inference import Inference
from bayes_vi.utils import to_ordered_dict

tfd = tfp.distributions
tfb = tfp.bijectors


class VI(Inference):
    """Implementation of Variational inference.

    Attributes
    ----------
    features: `tf.Tensor` or `dict[str, tf.Tensor]`
        The features contained in `dataset`.
    targets: `tf.Tensor`
        The targets contained in `dataset`.
    distribution: `tf.distributions.JointDistributionNamedAutoBatched`
        The joint distribution of the model (conditioned on features if regression model)
    target_log_prob: `callable`
        A callable taking a constrained parameter sample as a list
        and returning the unnormalized log posterior of the `model`.
    surrogate_posterior: `tfp.distributions.Distribution`
        A trainable distribution to fit to true posterior.
    discrepancy_fn: `callable`
        A function defining a f-divergence in log space.
    """

    def __init__(self, model, dataset, surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse):
        super(VI, self).__init__(model=model, dataset=dataset)

        self.features, self.targets = list(self.dataset.batch(self.num_datapoints).take(1))[0]
        self.targets = self.targets if self.model.is_generative_model else tf.expand_dims(self.targets, axis=0)

        self.distribution = model.get_joint_distribution(num_samples=self.targets.shape[0], features=self.features)
        self.surrogate_posterior = surrogate_posterior

        target_log_prob_fn = lambda state, targets: self.distribution.log_prob(
            **to_ordered_dict(self.model.param_names, state), y=targets
        )

        self.target_log_prob = self.surrogate_posterior.get_corrected_target_log_prob_fn(
            functools.partial(target_log_prob_fn, targets=self.targets)
        )
        self.discrepancy_fn = discrepancy_fn

    def safe_discrepancy_fn(self, logu):
        """Computes discrepancy, dropping infinite values"""
        discrepancy = self.discrepancy_fn(logu)
        return tf.boolean_mask(discrepancy, tf.math.is_finite(discrepancy))

    @staticmethod
    def make_optimizer_step_fn(loss_fn, optimizer):
        """Creates function that computes a single optimization step"""

        @tf.function(autograph=False)
        def optimizer_step():
            with tf.GradientTape() as tape:
                loss = loss_fn()

            grads = tape.gradient(loss, tape.watched_variables())
            clipped_grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]

            optimizer.apply_gradients(zip(clipped_grads, tape.watched_variables()))
            return loss

        return optimizer_step

    def fit(self, optimizer=tf.optimizers.Adam(), num_steps=10000, sample_size=1,
            num_samples_to_approx_marginals=10000, progress_bar=True):
        """Fits `surrogate_posterior` to the true posterior.

        Parameters
        ----------
        optimizer: `tf.optimizers.Optimizer`
            Optimizer instance to use for optimization (Default: `tf.optimizers.Adam()`).
        num_steps: `int`
            Number of optimization steps (Default: 10000).
        sample_size: `int`
            Number of samples to use in every optimization step to approximate variational loss (Default: 1).
        num_samples_to_approx_marginals: `int`
            Number of samples to use to finally approximate the marginal posterior distributions (Default: 10000).
        progress_bar: `bool`
            Boolean indicator whether or not to show a progress bar (Default: True).

        Returns
        -------
        approx_posterior: `tfp.distributions.JointDistributionNamedAutoBatched`
            Approximated marginal posterior distributions.
        losses: `np.array`
            Array of optimization loss history.
        """

        if progress_bar:
            steps = fastprogress.progress_bar(range(num_steps))
        else:
            steps = range(num_steps)

        losses = np.zeros(num_steps)

        def loss_fn():
            return tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn=self.target_log_prob,
                surrogate_posterior=self.surrogate_posterior.distribution,
                discrepancy_fn=self.safe_discrepancy_fn,
                use_reparameterization=True,
                sample_size=sample_size
            )

        optimizer_step = self.make_optimizer_step_fn(loss_fn, optimizer)

        for i in steps:
            loss = optimizer_step()
            losses[i] = loss
            if i % 10 == 0 and hasattr(steps, "comment"):
                steps.comment = "avg loss: {:.3f}".format(losses[max(0, i - 100):i + 1].mean())

        approx_posterior = self.surrogate_posterior.approx_joint_marginal_posteriors(num_samples_to_approx_marginals)
        return approx_posterior, losses

    @staticmethod
    def make_stochastic_optimizer_step_fn(loss_fn, optimizer):
        """Creates function that computes a single stochastic optimization step"""

        @tf.function(autograph=False)
        def stochastic_target_log_prob(log_prob_parts, dataset_size, batch_size):
            log_prob_parts_ = log_prob_parts.copy()
            data_log_prob = dataset_size * log_prob_parts_.pop('y') / batch_size,
            prior_log_prob = tf.math.add_n(list(log_prob_parts_.values()))
            return data_log_prob + prior_log_prob

        @tf.function(autograph=False)
        def optimizer_step(model, surrogate_posterior, features, targets, dataset_size, batch_size):
            targets = targets if model.is_generative_model else tf.expand_dims(targets, axis=0)
            distribution = model.get_joint_distribution(num_samples=targets.shape[0], features=features)
            target_log_prob_fn = lambda state, targets_: stochastic_target_log_prob(
                distribution.log_prob_parts(to_ordered_dict(model.param_names + ['y'], state + [targets_])),
                dataset_size,
                batch_size
            )
            target_log_prob = surrogate_posterior.get_corrected_target_log_prob_fn(
                functools.partial(target_log_prob_fn, targets_=targets)
            )
            with tf.GradientTape() as tape:
                loss = loss_fn(target_log_prob) / dataset_size

            grads = tape.gradient(loss, tape.watched_variables())
            clipped_grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]

            optimizer.apply_gradients(zip(clipped_grads, tape.watched_variables()))
            return loss

        return optimizer_step

    def stochastic_fit(self, optimizer=tf.optimizers.Adam(), epochs=1, batch_size=32, sample_size=1,
                       num_samples_to_approx_marginals=10000, progress_bar=True):
        """Fits `surrogate_posterior` to the true posterior stochastically.

        Note: This is particularly useful for large datasets.

        Parameters
        ----------
        optimizer: `tf.optimizers.Optimizer`
            Optimizer instance to use for optimization (Default: `tf.optimizers.Adam()`).
        epochs: `int`
            Number of full passes of the dataset (Default: 1).
        batch_size: `int`
            Number of examples from dataset to use on each optimization step (Default: 32).
        sample_size: `int`
            Number of samples to use in every optimization step to approximate variational loss (Default: 1).
        num_samples_to_approx_marginals: `int`
            Number of samples to use to finally approximate the marginal posterior distributions (Default: 10000).
        progress_bar: `bool`
            Boolean indicator whether or not to show a progress bar (Default: True).

        Returns
        -------
        approx_posterior: `tfp.distributions.JointDistributionNamedAutoBatched`
            Approximated marginal posterior distributions.
        losses: `np.array`
            Array of optimization loss history.
        """
        if progress_bar:
            epochs = fastprogress.master_bar(range(epochs))
        else:
            epochs = range(epochs)

        num_steps_per_epoch = self.num_datapoints // batch_size

        def loss_fn(target_log_prob_):
            return tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn=target_log_prob_,
                surrogate_posterior=self.surrogate_posterior.distribution,
                discrepancy_fn=self.safe_discrepancy_fn,
                use_reparameterization=True,
                sample_size=sample_size
            )

        optimizer_step = self.make_stochastic_optimizer_step_fn(loss_fn, optimizer)

        losses = []
        for _ in epochs:
            loss_epoch = np.zeros(shape=(num_steps_per_epoch,))
            data_generator = self.dataset.shuffle(batch_size * 100).batch(batch_size, drop_remainder=True)
            if progress_bar:
                batches = fastprogress.progress_bar(data_generator, parent=epochs)
            else:
                batches = data_generator

            for j, (features, targets) in enumerate(batches, start=1):
                loss = optimizer_step(
                    self.model,
                    self.surrogate_posterior,
                    features,
                    targets,
                    self.num_datapoints,
                    batch_size
                )
                loss_epoch[j - 1] = loss
                if j % 10 == 0 and hasattr(batches, "comment"):
                    epochs.child.comment = "avg loss: {:.3f}".format(loss_epoch[max(0, j - 100):j + 1].mean())
            epochs.main_bar.comment = "avg loss: {:.3f}".format(loss_epoch.mean())

            losses.append(loss_epoch)
        approx_posterior = self.surrogate_posterior.approx_joint_marginal_posteriors(num_samples_to_approx_marginals)
        return approx_posterior, np.concatenate(losses, axis=0)
