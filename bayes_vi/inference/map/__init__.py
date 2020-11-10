import tensorflow as tf

from bayes_vi.inference import Inference


class MAP(Inference):

    def __init__(self, model: Model, dataset: tf.data.Dataset):
        super(MAP, self).__init__(model=model, dataset=dataset)
        self.model = model
        self.dataset = dataset

    def fit(self):
        pass
