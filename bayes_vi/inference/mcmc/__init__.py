import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp
from fastprogress import fastprogress

from bayes_vi.inference import Inference
from bayes_vi.inference.mcmc.stepsize_adaptation_kernels import StepSizeAdaptationKernel
from bayes_vi.inference.mcmc.transition_kernels import TransitionKernel, RandomWalkMetropolis
from bayes_vi.models import Model
from bayes_vi.utils import compose
from bayes_vi.utils import to_ordered_dict
from bayes_vi.utils.sample_results import SampleResult

tfd = tfp.distributions
tfb = tfp.bijectors


class MCMC(Inference):
    """Implementation of MCMC to generate posterior samples.

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
        and returning the unnormalized log posterior of the `model`.
    transition_kernel: `bayes_vi.inference.mcmc.transition_kernels.TransitionKernel`
        A Markov transition kernel to define transition between states.
        (Default: `bayes_vi.inference.mcmc.transition_kernels.RandomWalkMetropolis`).
    """

    def __init__(self, model, dataset, transition_kernel=RandomWalkMetropolis()):
        """Initializes MCMC.

        model: `Model`
            A Bayesian probabilistic `Model`.
        dataset: `tf.data.Dataset`
            A `tf.data.Dataset` consisting of features (if regression model) and targets.
        transition_kernel: `bayes_vi.inference.mcmc.transition_kernels.TransitionKernel`
            A Markov transition kernel to define transition between states.
            (Default: `bayes_vi.inference.mcmc.transition_kernels.RandomWalkMetropolis`).
        """
        super(MCMC, self).__init__(model=model, dataset=dataset)
        # take num_datapoints as a single batch and extract features and targets
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]
        self.targets = self.targets if model.is_generative_model else tf.expand_dims(self.targets, axis=0)

        # condition model on features
        self.model = model
        self.distribution = model.get_joint_distribution(targets=self.targets, features=self.features)

        self.target_log_prob_fn = lambda state, targets: self.distribution.log_prob(
                **to_ordered_dict(model.param_names, state), y=targets
            )

        # define target_log_prob_fn as the log probability of the model, being the unnormalized log posterior
        self.target_log_prob = tf.function(
            functools.partial(self.target_log_prob_fn, targets=self.targets)
        )

        self.transition_kernel = transition_kernel

    def fit(self, initial_state=None, num_chains=4, num_samples=4000,
            num_burnin_steps=1000, merge_state_parts=False, progress_bar=False):
        """Fits the Bayesian model to the dataset.

        Parameters
        ----------
        initial_state: `collection.OrderedDict[str, tf.Tensor]`
            A parameter sample of some sample shape, where the sample shape induces parallel runs.
            Or an explicitly specified initial state of the shape of a parameter sample.
            Special values:
                - `None`: uses prior samples .
                - `ones`: initializes all parameters to 1. (disturbed by `jitter`)
            (Default: `None`).
        num_chains: `int`
            Number of chains to run in parallel. (Default: `4`).
        num_samples: `int`
            Number of samples to generate per chain. (Default: `4000`).
        num_burnin_steps: `int`
            Number of burnin steps before starting to generate `num_samples`. (Default: `1000`).
        merge_state_parts: `bool`
            A boolean indicator whether or not to merge the state parts into a single state.
            Is set to `True` if a single `tfp.bijectors.Bijector` is provided as `transforming_bijectors`.
            (Default: `False`).

        Returns
        -------
        `bayes_vi.utils.sample_results.SampleResult`
            The generated posterior samples. Provides precomputed sample statistics.
        """
        # generate initial state for markov chain
        if initial_state is None:
            # sample uniform [-2,2] in unconstrained space
            uniform_sample = [
                tf.random.uniform([num_chains] + shape, minval=-1., maxval=1.)
                for shape in self.model.unconstrained_param_event_shape
            ]
            initial_state = self.model.joint_constraining_bijector.forward(uniform_sample)
        elif isinstance(initial_state, (list, collections.OrderedDict)):
            # define initial state based on given OrderedDict with explicit values for each parameter
            initial_state = initial_state if isinstance(initial_state, list) else list(initial_state.values())
        else:
            raise ValueError("`initial_state` has to be in [None, list, collections.OrderedDict]; "
                             "was {}".foramt(initial_state))

        # components of `initial_state` are considered independent
        # if transforming_bijectors is a single bijector or merge_state_parts is explicitly set,
        # we construct a single component initial state, such that all parameters are considered dependent
        if hasattr(self.transition_kernel, 'transforming_bijector') and \
                isinstance(self.transition_kernel.transforming_bijector, tfb.Bijector):
            merge_state_parts = True

        if merge_state_parts:
            initial_state = self.model.split_flat_param_bijector.inverse(
                self.model.reshape_flat_param_bijector.inverse(initial_state)
            )
            constraining_bijector = self.model.blockwise_constraining_bijector
            target_log_prob = lambda *state: self.target_log_prob(
                self.model.reshape_flat_param_bijector.forward(
                    self.model.split_flat_param_bijector.forward(state[0])
                )
            )
        else:
            constraining_bijector = self.model.constraining_bijectors
            target_log_prob = lambda *state: self.target_log_prob(state)

        # build given Markov transition kernel
        kernel, trace_fns, trace_metrics = self.transition_kernel(target_log_prob)

        # wrap transition kernel to generate markov chains in unconstrained space
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=kernel,
            bijector=constraining_bijector
        )
        trace_fns.append(lambda _, pkr: (_, pkr.inner_results))

        if progress_bar:
            pbar = tfp.experimental.mcmc.ProgressBarReducer(num_samples, progress_bar_fn=lambda n: iter(
                fastprogress.progress_bar(range(n))))
            kernel = tfp.experimental.mcmc.WithReductions(kernel, pbar)
            trace_fns.append(lambda _, pkr: (_, pkr.inner_results))

        # define `trace_fn` for Markov chain as composition of all defined trace functions
        trace_fn = compose(list(reversed(trace_fns)))

        # apply `tfp.mcmc.sample_chain` with a convenience wrapper
        # to allow for annotation with `tf.function` giving performance boost
        samples, trace = self.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=trace_fn
        )

        if merge_state_parts:
            samples = self.model.reshape_flat_param_bijector.forward(
                self.model.split_flat_param_bijector.forward(samples)
            )
        trace_dict = {name: values[0] if isinstance(values, list) else values for name, values in
                      zip(trace_metrics, trace)}

        # return `SampleResult` object containing samples, trace
        return SampleResult(samples, trace_dict)

    @staticmethod
    @tf.function(autograph=False)
    def sample_chain(num_results, num_burnin_steps, current_state, kernel, trace_fn):
        """wraps `tfp.mcmc.sample_chain`."""
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=current_state,
            kernel=kernel,
            trace_fn=trace_fn
        )
