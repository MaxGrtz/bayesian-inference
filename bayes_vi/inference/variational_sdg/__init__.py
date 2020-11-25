from copy import deepcopy

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.utils.sample_results import SampleResult

tfb = tfp.bijectors


class VariationalSGD(Inference):
    """Implementation of VariationalSGD to approximate posterior samples.

    Note: The approach is based on the observation that SGD (stochastic gradient descent)
          with constant learning rate corresponds to a Markov chain. This allows us to
          scale posterior sampling to larger datasets via batched SGD.

    Attributes
    ----------
    model: `Model`
        A Bayesian probabilistic `Model`.
    dataset: `tf.data.Dataset`
        A `tf.data.Dataset` consisting of features (if regression model) and targets.
    state: `tf.Variable`
        A trainable variable initialized with `initial_state` (parameter sample of some sample shape)
        in `fit` method.
    optimizer: `tfp.optimizer.VariationalSGD`
        Optimizer used in each `train_step`.
    batch_size: `int`
        Number of examples (i.e. datapoints) considered in each `train_step`.
    data_batch_ratio: `int`
        The number of batches in the dataset (num_examples // batch_size).
        Needed to scale down prior log prob component in loss computation (with 1/batch_data_ratio).
    recorded_states: `list` of `tf.Tensor`
        accumulates the value of `state` in every optimization step.
    """

    def __init__(self, model, dataset):
        """Initializes VariationalSGD.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        dataset: `tf.data.Dataset`
            A `tf.data.Dataset` consisting of features (if regression model) and targets.
        """
        super(VariationalSGD, self).__init__(model=model, dataset=dataset)
        self.state = None
        self.optimizer = None
        self.batch_size = None
        self.data_batch_ratio = None
        self.recorded_states = []

    def fit(self, initial_state, max_learning_rate=1.0, preconditioner_decay_rate=0.95,
            burnin=25, burnin_max_learning_rate=1e-6, batch_size=25, repeat=1,
            shuffle=1000, epochs=100):
        """Fits the Bayesian model to the dataset.

        Parameters
        ----------
        initial_state: `collection.OrderedDict[str, tf.Tensor]`
            A parameter sample of some sample shape, where the sample shape induces parallel runs.
            Or an explicitly specified initial state of the shape of a parameter sample.
        max_learning_rate: `float`
            A maximum allowable effective coordinate-wise learning rate. The algorithm scales down any
            effective learning rate (i.e. after preconditioning) that is larger than this. (Default: `1.0`)
        preconditioner_decay_rate: `float`
            The exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior. (Default: `0.95`)
        burnin: `int`
            The number of iterations to collect gradient statistics to update the preconditioner
            before starting to draw noisy samples. (Default: `25`)
        burnin_max_learning_rate: `float`
            Maximum learning rate to use during the burnin period. (Default: `1e-6`)
        batch_size: `int`
            Number of examples (i.e. datapoints) considered in each `train_step`. (Default: `25`).
        repeat: `int`
            Number of times to duplicate the dataset. (Default: `1`).
        shuffle: `int`
            Number of samples to consider for shuffling. (Default: `1000`).
        epochs: `int`
            Number of epochs to train for, i.e. how often to iterate over complete dataset. (Default: `100`).

        Returns
        -------
        losses: `list` of `tf.Tensor`
            Training `loss` on each optimization step.
        final_state: `collections.OrderedDict[str, tf.Tensor]`
            The final state of the optimization, having the same shape as `initial_state`.
        sample_results: `bayes_vi.utils.sample_results.SampleResult`
            The generated posterior samples. Provides precomputed sample statistics.
        """
        self.state = tf.Variable(self.unconstrain_flatten_and_merge(initial_state))
        self.batch_size = batch_size
        self.data_batch_ratio = self.num_examples // self.batch_size
        self.optimizer = tfp.optimizer.VariationalSGD(batch_size=self.batch_size,
                                                      total_num_examples=self.num_examples,
                                                      max_learning_rate=max_learning_rate,
                                                      preconditioner_decay_rate=preconditioner_decay_rate,
                                                      burnin=burnin,
                                                      burnin_max_learning_rate=burnin_max_learning_rate)

        losses = self.training(batch_size=self.batch_size, repeat=repeat, shuffle=shuffle, epochs=epochs)

        final_state = self.split_reshape_constrain_and_to_dict(self.state)
        samples = list(self.split_reshape_constrain_and_to_dict(tf.stack(self.recorded_states[burnin:])).values())
        sample_results = SampleResult(model=self.model, samples=samples, trace=None)
        return losses, final_state, sample_results

    @tf.function
    def loss(self, state, y):
        """Computes the negative unnormalized log posterior of the model for the given state.

        Note: the prior contribution is scaled down based on `data_batch_ratio` to
              allow for consistent stochastic gradient descent solution.

        Parameters
        ----------
        state: `tf.Variable`
            State (parameter values) for which to evaluate loss.
        y: `tf.Tensor`
            Target batch on which to evaluate loss.

        Returns
        -------
        `tf.Tensor`
            Training loss on current iteration.
        """
        prior_log_prob, data_log_prob = self.model.unnormalized_log_posterior_parts(
            self.split_reshape_constrain_and_to_dict(self.state), y)
        jacobian_det_correction = self.model.target_log_prob_correction_forward(state)
        return - (prior_log_prob + jacobian_det_correction) / self.data_batch_ratio \
               - data_log_prob

    def training(self, batch_size, repeat, shuffle, epochs):
        """Main optimization loop.

        Parameters
        ----------
        batch_size: `int`
            Number of examples (i.e. datapoints) considered in each `train_step`. (Default: `25`).
        repeat: `int`
            Number of times to duplicate the dataset. (Default: `1`).
        shuffle: `int`
            Number of samples to consider for shuffling. (Default: `1000`).
        epochs: `int`
            Number of epochs to train for, i.e. how often to iterate over complete dataset. (Default: `100`).

        Returns
        -------
        losses: `list` of `tf.Tensor`
            Training `loss` on each iteration.
        """
        losses = []
        for epoch in range(1, epochs + 1):
            for x, y in self.dataset.repeat(repeat).shuffle(shuffle).batch(batch_size):
                self.recorded_states.append(deepcopy(self.state))
                self.model = self.model(x)
                loss = self.train_step(y)
                losses.append(loss)
        return losses

    @tf.function
    def train_step(self, y):
        """Single optimization step.

        Parameters
        ----------
        y: `tf.Tensor`
            Target batch on which to compute and apply gradients.

        Returns
        -------
        `tf.Tensor`
            Training loss on current iteration.
        """
        with tf.GradientTape() as tape:
            loss = self.loss(self.state, y)
        # Compute and apply gradients
        trainable_vars = tape.watched_variables()
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss
