import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.utils import make_transformed_log_prob, to_ordered_dict
from bayes_vi.inference.mcmc.sample_results import SampleResult

tfb = tfp.bijectors


class SGD(Inference):

    def __init__(self, model, dataset):
        super(SGD, self).__init__(model=model, dataset=dataset)
        self.state = None
        self.num_examples = int(self.dataset.cardinality())
        self.data_batch_ratio = None
        self.recorded_states = []
        self.optimizer = None
        unnormalized_log_posterior = lambda state, y: self.model.unnormalized_log_posterior(
            to_ordered_dict(self.model.param_names, state), y)
        self.transformed_log_prob = make_transformed_log_prob(unnormalized_log_posterior,
                                                              bijector=self.model.reshape_constrain_bijectors,
                                                              direction='forward')

    def fit(self, initial_state, burnin=25, batch_size=25, repeat=1, shuffle=1000, epochs=10, **optimizer_kwargs):
        self.state = tf.Variable(self.model.transform_state_inverse(initial_state))
        self.data_batch_ratio = self.num_examples // batch_size
        self.optimizer = tfp.optimizer.VariationalSGD(batch_size,
                                                      self.num_examples,
                                                      burnin=burnin,
                                                      **optimizer_kwargs)
        losses = self.training(batch_size=batch_size, repeat=repeat, shuffle=shuffle, epochs=epochs)

        final_state = self.model.transform_state_forward(self.state)
        samples = [tf.stack(param[burnin:]) for param in map(list, zip(*self.recorded_states))]
        # reduce one dimensional params to scalar
        samples = [tf.reshape(s, shape=s.shape[:-1]) if s.shape[-1] == 1 else s for s in samples]
        sample_results = SampleResult(model=self.model, samples=samples, trace=None)
        return losses, final_state, sample_results

    def loss(self, state, y):
        return - self.transformed_log_prob(self.model.split_bijector(state), y)

    def training(self, batch_size, repeat, shuffle, epochs):
        losses = []
        for epoch in range(1, epochs + 1):
            for x, y in self.dataset.repeat(repeat).shuffle(shuffle).batch(batch_size):
                self.recorded_states.append(
                    list(self.model.transform_state_forward(self.state).values())
                )
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
