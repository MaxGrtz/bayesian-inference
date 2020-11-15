import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.utils import make_transformed_log_prob, make_val_and_grad_fn, to_ordered_dict

tfb = tfp.bijectors


class PointEstimate(Inference):

    def __init__(self, model, dataset):
        super(PointEstimate, self).__init__(model=model, dataset=dataset)
        self.state = None
        self.optimizer = None

    def fit(self, initial_state, optimizer, batch_size=25, repeat=1, shuffle=1000, epochs=10):
        self.state = tf.Variable(self.model.transform_state_inverse(initial_state))
        self.optimizer = optimizer
        losses = self.training(batch_size=batch_size, repeat=repeat, shuffle=shuffle, epochs=epochs)
        return losses, self.model.transform_state_forward(self.state)

    def loss(self, state, y):
        raise NotImplementedError("No loss implemented.")

    def training(self, batch_size, repeat, shuffle, epochs):
        losses = []
        for epoch in range(1, epochs + 1):
            for x, y in self.dataset.repeat(repeat).shuffle(shuffle).batch(batch_size):
                self.model = self.model(x)
                loss = self.train_step(y)
                losses.append(loss)
        return losses

    @tf.function
    def train_step(self, y):
        with tf.GradientTape() as tape:
            loss = self.loss(self.state, y)
        # Compute and apply gradients
        trainable_vars = tape.watched_variables()
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss


class MLE(PointEstimate):

    def __init__(self, model, dataset):
        super(MLE, self).__init__(model=model, dataset=dataset)

    @tf.function
    def loss(self, state, y):
        return - self.model.log_likelihood(self.model.transform_state_forward(state), y)


class MAP(PointEstimate):

    def __init__(self, model, dataset):
        super(MAP, self).__init__(model=model, dataset=dataset)
        unnormalized_log_posterior = lambda state, y: self.model.unnormalized_log_posterior(
            to_ordered_dict(self.model.param_names, state), y)
        self.transformed_log_prob = make_transformed_log_prob(unnormalized_log_posterior,
                                                              bijector=self.model.reshape_constrain_bijectors,
                                                              direction='forward')

    @tf.function
    def loss(self, state, y):
        return - self.transformed_log_prob(self.model.split_bijector(state), y)


class BFGS(Inference):

    def __init__(self, model, dataset):
        super(BFGS, self).__init__(model=model, dataset=dataset)

    def fit(self, initial_state, target_log_prob, num_parallel_runs=1, jitter=0.1, save_memory=True, **optimizer_kwargs):
        unconstrained_state = self.model.transform_state_inverse(initial_state)
        initial_pos = [tf.random.normal(unconstrained_state.shape, mean=unconstrained_state, stddev=jitter)
                       if float(jitter) != 0. else unconstrained_state for _ in range(num_parallel_runs)]

        loss = lambda state: - target_log_prob(self.model.transform_state_forward(state))

        if save_memory:
            res = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function=make_val_and_grad_fn(loss),
                initial_position=initial_pos,
                **optimizer_kwargs
            )
        else:
            res = tfp.optimizer.bfgs_minimize(
                value_and_gradients_function=make_val_and_grad_fn(loss),
                initial_position=initial_pos,
                **optimizer_kwargs
            )

        return res, \
               [self.model.transform_state_forward(x) for x in res.position[res.converged]], \
               [self.model.transform_state_forward(x) for x in res.position[~res.converged]]
