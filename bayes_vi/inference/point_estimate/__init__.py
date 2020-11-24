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
        self.num_examples = int(self.dataset.cardinality())
        self.batch_size = None
        self.data_batch_ratio = None
        self.unconstrain_flatten_and_merge = lambda state: self.model.split_unconstrained_bijector.inverse(
            self.model.flatten_unconstrained_sample(
                self.model.unconstrain_sample(state.values())
            )
        )
        self.split_reshape_constrain_and_to_dict = lambda state: to_ordered_dict(
            self.model.param_names,
            self.model.constrain_sample(
                self.model.reshape_flat_unconstrained_sample(
                    self.model.split_unconstrained_bijector.forward(state)
                )
            )
        )

    def fit(self, initial_state, optimizer, batch_size=25, repeat=1, shuffle=1000, epochs=10):
        self.state = tf.Variable(self.unconstrain_flatten_and_merge(initial_state))
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.data_batch_ratio = self.num_examples // self.batch_size
        losses = self.training(batch_size=self.batch_size, repeat=repeat, shuffle=shuffle, epochs=epochs)
        return losses, self.split_reshape_constrain_and_to_dict(self.state)

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
        return - self.model.log_likelihood(self.split_reshape_constrain_and_to_dict(self.state), y)


class MAP(PointEstimate):

    def __init__(self, model, dataset):
        super(MAP, self).__init__(model=model, dataset=dataset)

    @tf.function
    def loss(self, state, y):
        prior_log_prob, data_log_prob = self.model.unnormalized_log_posterior_parts(
            self.split_reshape_constrain_and_to_dict(self.state), y)
        jacobian_det_correction = self.model.target_log_prob_correction_forward(state)
        return - (prior_log_prob + jacobian_det_correction) / self.data_batch_ratio \
               - data_log_prob


class BFGS(PointEstimate):

    def __init__(self, model, dataset):
        super(BFGS, self).__init__(model=model, dataset=dataset)
        self.target_log_prob = None
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]
        self.model = model(features=self.features)

    def fit(self, initial_state, target_log_prob, limited_memory=False, **optimizer_kwargs):
        initial_state = self.unconstrain_flatten_and_merge(initial_state)
        self.target_log_prob = target_log_prob

        if limited_memory:
            res = self.lbfgs_minimize(self.loss, initial_state, **optimizer_kwargs)
        else:
            res = self.bfgs_minimize(self.loss, initial_state, **optimizer_kwargs)

        return res, \
               self.split_reshape_constrain_and_to_dict(res.position[res.converged]), \
               self.split_reshape_constrain_and_to_dict(res.position[~res.converged])

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
