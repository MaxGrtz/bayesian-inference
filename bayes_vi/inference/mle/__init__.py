import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.utils.bijectors import Mapper
from bayes_vi.utils.functions import make_val_and_grad_fn


class MLE(Inference):

    def __init__(self, model, dataset):
        super(MLE, self).__init__(model=model, dataset=dataset)


    def fit(self):
        pass
        # batch = list(self.dataset.batch(dataset.cardinality()).take(1))[0]
        # try:
        #     features, targets = batch
        # except ValueError:
        #     features, targets = None, batch
        #
        # model = self.model(features, targets)
        #
        # initial_state = model.distribution.sample(1).values()[:-1]
        # event_shape = model.distribution.event_shape[:-1]
        #
        # mapper = Mapper(initial_state, model.constraining_bijectors, event_shape)
        #
        # reshape_add_targets = lambda state, y: collections.OrderedDict(
        #     **collections.OrderedDict([(k, v) for k, v
        #                                in zip(model.param_names, mapper.split_and_reshape(state))]),
        #     y=model.targets
        # )
        # neg_log_likelihood = lambda x: - model.distribution.log_prob_parts(reshape_add_targets(x))['y']
        #
        # lbfgs_results = tfp.optimizer.lbfgs_minimize(
        #     neg_log_likelihood,
        #     initial_position=mapper.flatten_and_concat(initial_state),
        #     tolerance=1e-20,
        #     x_tolerance=1e-8
        # )
        # return lbfgs_results


