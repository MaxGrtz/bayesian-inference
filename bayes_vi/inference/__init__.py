import tensorflow as tf

from bayes_vi.models import Model


class Inference:
    """A base class for different Inference algorithms.

    An `Inference` algorithm consists of:
        a Bayesian `Model` and
        a `tf.data.Dataset`.

    Note: Every subclass has to implement a `fit` method.

    Attributes
    ----------
    model: `Model`
        A Bayesian probabilistic `Model`.
    dataset: `tf.data.Dataset`
        A `tf.data.Dataset` consisting of features (if regression model) and targets.
    """

    def __init__(self, model, dataset):
        """Initializes an Inference instance.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        dataset: `tf.data.Dataset`
            A `tf.data.Dataset` consisting of features (if regression model) and targets.
        """
        self.model = model
        self.dataset = dataset
        self.num_examples = int(self.dataset.cardinality())
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
