import tensorflow as tf

from bayes_vi.models import Model


class Inference:
    """A base class for different Inference algorithms.

    An `Inference` algorithm consists of:
        a Bayesian `Model` and
        a `tf.data.Dataset`.

    Attributes
    ----------
    model: `Model`
        A Bayesian probabilistic `Model`.
    dataset: `tf.data.Dataset`
        A `tf.data.Dataset` consisting of features (if regression model) and targets.
    """

    def __init__(self, model, dataset):
        """Initializes and Inference instance.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        dataset: `tf.data.Dataset`
            A `tf.data.Dataset` consisting of features (if regression model) and targets.
        """
        self.model = model
        self.dataset = dataset

    def fit(self, *args, **kwargs):
        """Fits the Bayesian model to the dataset.

        Has to be implemented in any subclass of `Inference`.

        Parameters
        ----------
        args: positional arguments
            positional arguments of the fit method.
        kwargs: keyword arguments
            keyword arguments of the fit method.

        Raises
        ------
        NotImplementedError
            If the fit method is not implemented in a subclass.
        """
        raise NotImplementedError('fit is not yet implemented!')
