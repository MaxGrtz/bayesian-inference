import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.utils import make_val_and_grad_fn

tfb = tfp.bijectors


class PointEstimate(Inference):
    """Base class for point estimate algorithms to infer model parameters.

    Attributes
    ----------
    model: `Model`
        A Bayesian probabilistic `Model`.
    dataset: `tf.data.Dataset`
        A `tf.data.Dataset` consisting of features (if regression model) and targets.
    state: `tf.Variable`
        A trainable variable initialized with `initial_state` (parameter sample of some sample shape)
        in `fit` method.
    optimizer: `tf.optimizers.Optimizer`
        Optimizer to use in each `train_step`.
    batch_size: `int`
        Number of examples (i.e. datapoints) considered in each `train_step`.
    data_batch_ratio: `int`
        The number of batches in the dataset (num_examples // batch_size).
        Needed to scale down prior log prob component in loss computation (with 1/batch_data_ratio).
    """

    def __init__(self, model, dataset):
        """Initializes PointEstimate instance.

        Note: This is only a base class, subclasses have to implement a loss function.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        dataset: `tf.data.Dataset`
            A `tf.data.Dataset` consisting of features (if regression model) and targets.
        """
        super(PointEstimate, self).__init__(model=model, dataset=dataset)
        self.state = None
        self.optimizer = None
        self.batch_size = None
        self.data_batch_ratio = None

    def fit(self, initial_state, optimizer, batch_size=25, repeat=1, shuffle=1000, epochs=100):
        """Fits the Bayesian model to the dataset.

        Parameters
        ----------
        initial_state: `collection.OrderedDict[str, tf.Tensor]`
            A parameter sample of some sample shape, where the sample shape induces parallel runs.
            Or an explicitly specified initial state of the shape of a parameter sample.
        optimizer: `tf.optimizers.Optimizer`
            Optimizer to use in each `train_step`.
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
        """
        self.state = tf.Variable(self.unconstrain_flatten_and_merge(initial_state))
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.data_batch_ratio = self.num_examples // self.batch_size
        losses = self.training(batch_size=self.batch_size, repeat=repeat, shuffle=shuffle, epochs=epochs)
        final_state = self.split_reshape_constrain_and_to_dict(self.state)
        return losses, final_state

    def loss(self, state, y):
        """Computes the optimization loss for a given state and target batch y.

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
        raise NotImplementedError("No loss implemented.")

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
                self.model(x)
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


class MLE(PointEstimate):
    """Implementation of Maximum Likelihood Estimate (MLE) for model parameters."""

    def __init__(self, model, dataset):
        super(MLE, self).__init__(model=model, dataset=dataset)

    @tf.function
    def loss(self, state, y):
        """Computes the negative log likelihood of the model for the given state."""
        return - self.model.log_likelihood(self.split_reshape_constrain_and_to_dict(self.state), y)


class MAP(PointEstimate):
    """Implementation of Maximum a Posteriori (MAP) estimate for model parameters."""

    def __init__(self, model, dataset):
        super(MAP, self).__init__(model=model, dataset=dataset)

    @tf.function
    def loss(self, state, y):
        """Computes the negative unnormalized log posterior of the model for the given state.

        Note: the prior contribution is scaled down based on `data_batch_ratio` to
              allow for consistent stochastic gradient descent solution.
        """
        prior_log_prob, data_log_prob = self.model.unnormalized_log_posterior_parts(
            self.split_reshape_constrain_and_to_dict(self.state), y)
        jacobian_det_correction = self.model.target_log_prob_correction_forward(state)
        return - (prior_log_prob + jacobian_det_correction) / self.data_batch_ratio \
               - data_log_prob


class BFGS(Inference):
    """BFGS optimizer class model parameter point estimates.

    Note: This wraps `tfp.optimizer.bfgs_minimize`.

    Attributes
    ----------
    model: `Model`
        A Bayesian probabilistic `Model`.
    dataset: `tf.data.Dataset`
        A `tf.data.Dataset` consisting of features (if regression model) and targets.
    features: `tf.Tensor` or `dict[str, tf.Tensor]`
        The features contained in `dataset`.
    targets: `tf.Tensor`
        The targets contained in `dataset`.
    target_log_prob: `callable`
        A callable taking a constrained parameter sample of some sample shape
        and returning log probabilities.
    """

    def __init__(self, model, dataset, target_log_prob):
        """Initializes the BFGS optimizer.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        dataset: `tf.data.Dataset`
            A `tf.data.Dataset` consisting of features (if regression model) and targets.
        target_log_prob: `callable`
            A callable taking a constrained parameter sample of some sample shape
            and returning log probabilities.
        """
        super(BFGS, self).__init__(model=model, dataset=dataset)
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]
        self.model(features=self.features)
        self.target_log_prob = target_log_prob

    def fit(self, initial_state, limited_memory=False, **optimizer_kwargs):
        """Fits the Bayesian model to the dataset.

        Parameters
        ----------
        initial_state: `collection.OrderedDict[str, tf.Tensor]`
            A parameter sample of some sample shape, where the sample shape induces parallel runs.
            Or an explicitly specified initial state of the shape of a parameter sample.
        limited_memory: `bool`
            A boolean indicator whether to use limited memory version of the optimizer, i.e.
            `tfp.optimizer.lbfgs_minimize`.
        **optimizer_kwargs: keyword arguments
            Additional optimizer keyword arguments.

        Returns
        -------
        optimizer_results: `collections.namedtuple`
            A namedtuple of the optimizer results.
        converged_states: `collections.OrderedDict[str, tf.Tensor]`
            An ordered dict containing the parameter values of the converged states.
        diverged_states: `collections.OrderedDict[str, tf.Tensor]`
            An ordered dict containing the parameter values of the diverged states.
        """
        initial_state = self.unconstrain_flatten_and_merge(initial_state)

        if limited_memory:
            optimizer_results = self.lbfgs_minimize(self.loss, initial_state, **optimizer_kwargs)
        else:
            optimizer_results = self.bfgs_minimize(self.loss, initial_state, **optimizer_kwargs)
        converged_states = self.split_reshape_constrain_and_to_dict(
            optimizer_results.position[optimizer_results.converged]
        )
        diverged_states = self.split_reshape_constrain_and_to_dict(
            optimizer_results.position[~optimizer_results.converged]
        )
        return optimizer_results, converged_states, diverged_states

    @tf.function
    def loss(self, state):
        return - self.target_log_prob(self.split_reshape_constrain_and_to_dict(state))

    @tf.function
    def lbfgs_minimize(self, loss, initial_state, **optimizer_kwargs):
        return tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=make_val_and_grad_fn(loss),
            initial_position=initial_state,
            **optimizer_kwargs
        )

    @tf.function
    def bfgs_minimize(self, loss, initial_state, **optimizer_kwargs):
        return tfp.optimizer.bfgs_minimize(
            value_and_gradients_function=make_val_and_grad_fn(loss),
            initial_position=initial_state,
            **optimizer_kwargs
        )
