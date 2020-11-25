import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp

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
    step_size_adaptation_kernel: `bayes_vi.inference.mcmc.stepsize_adaptation_kernels.StepSizeAdaptationKernel`
        A stepsize adaptation kernel to wrap the transition kernel and optimize stepsize in burnin phase.
        (Default: `None`)
    transforming_bijectors: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
        A single or per state part transforming bijector to transform the generated samples.
        This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
        the target distribution for more efficient sampling.
        In the context of HMC this is approximately Riemannian-HMC (RHMC).
    """

    def __init__(self, model, dataset, transition_kernel=RandomWalkMetropolis(),
                 step_size_adaptation_kernel=None, transforming_bijectors=None):
        """Initializes MCMC.

        model: `Model`
            A Bayesian probabilistic `Model`.
        dataset: `tf.data.Dataset`
            A `tf.data.Dataset` consisting of features (if regression model) and targets.
        transition_kernel: `bayes_vi.inference.mcmc.transition_kernels.TransitionKernel`
            A Markov transition kernel to define transition between states.
            (Default: `bayes_vi.inference.mcmc.transition_kernels.RandomWalkMetropolis`).
        step_size_adaptation_kernel: `bayes_vi.inference.mcmc.stepsize_adaptation_kernels.StepSizeAdaptationKernel`
            A stepsize adaptation kernel to wrap the transition kernel and optimize stepsize in burnin phase.
            (Default: `None`)
        transforming_bijectors: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
            A single or per state part transforming bijector to transform the generated samples.
            This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
            the target distribution for more efficient sampling.
            In the context of HMC this is approximately Riemannian-HMC (RHMC).
        """
        super(MCMC, self).__init__(model=model, dataset=dataset)
        # take num_datapoints as a single batch and extract features and targets
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]

        # condition model on features
        self.model = model(features=self.features)

        # define target_log_prob_fn as the log probability of the model, being the unnormalized log posterior
        self.target_log_prob = functools.partial(self.model.unnormalized_log_posterior, targets=self.targets)

        self.transition_kernel = transition_kernel
        self.step_size_adaptation_kernel = step_size_adaptation_kernel
        self.transforming_bijectors = transforming_bijectors

    def fit(self, initial_state=None, jitter=0.0001, num_chains=4, num_samples=4000,
            num_burnin_steps=1000, merge_state_parts=False):
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
        jitter: `float`
            Stddev with which to disturb `initial_state`, in the case of `initial_state='ones'`.
            (Default: `0.0001`).
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
            # sample initial state from priors
            initial_state = list(self.model.prior_distribution.sample(num_chains).values())
        elif initial_state == 'ones':
            # set initial state to all ones
            sample = [tf.ones_like(x) for x in self.model.prior_distribution.sample().values()]
            initial_state = [
                tf.random.normal([num_chains] + part.shape, mean=part, stddev=max(0.01, float(jitter)))
                for part in sample
            ]
        elif isinstance(initial_state, collections.OrderedDict):
            # define initial state based on given OrderedDict with explicit values for each parameter
            initial_state = list(initial_state.values())
        else:
            raise ValueError("`initial_state` has to be in [None, 'ones', collections.OrderedDict]; "
                             "was {}".foramt(initial_state))

        # components of `initial_state` are considered independent
        # if transforming_bijectors is a single bijector or merge_state_parts is explicitly set,
        # we construct a single component initial state, such that all parameters are considered dependent
        if isinstance(self.transforming_bijectors, tfb.Bijector):
            merge_state_parts = True

        if merge_state_parts:
            initial_state = self.model.split_constrained_bijector.inverse(
                self.model.flatten_constrained_sample(initial_state)
            )
            constraining_bijector = self.model.blockwise_constraining_bijector
            target_log_prob_fn = lambda *state: self.target_log_prob(
                to_ordered_dict(
                    self.model.param_names,
                    self.model.reshape_flat_constrained_sample(self.model.split_constrained_bijector(state[0]))
                )
            )
        else:
            constraining_bijector = self.model.constraining_bijectors
            target_log_prob_fn = lambda *state: self.target_log_prob(
                to_ordered_dict(self.model.param_names, state)
            )

        # build given Markov transition kernel
        kernel = self.transition_kernel(target_log_prob_fn)
        # add transition kernels trace_fn to list of trace functions (for later composition)
        trace_fns = [self.transition_kernel.trace_fn]

        # wrap `TransitionKernel` in a `StepSizeAdaptationKernel` if one is provided
        if self.step_size_adaptation_kernel:
            kernel = self.step_size_adaptation_kernel(inner_kernel=kernel)
            # append another trace function reaching through to the inner kernel
            trace_fns.append(self.step_size_adaptation_kernel.trace_fn)

        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=kernel,
            bijector=constraining_bijector
        )
        trace_fns.append(lambda _, pkr: (_, pkr.inner_results))

        if self.transforming_bijectors:
            # wrap in another `TransformedTransitionKernel`,
            kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=kernel,
                bijector=self.transforming_bijectors
            )
            # append another trace function reaching through to the inner kernel
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
            samples = self.model.reshape_flat_constrained_sample(
                self.model.split_constrained_bijector.forward(samples)
            )

        # return `SampleResult` object containing samples, trace and statistics
        return SampleResult(self.model, samples, trace)

    @staticmethod
    @tf.function
    def sample_chain(num_results, num_burnin_steps, current_state, kernel, trace_fn):
        """wraps `tfp.mcmc.sample_chain`."""
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=current_state,
            kernel=kernel,
            trace_fn=trace_fn
        )
