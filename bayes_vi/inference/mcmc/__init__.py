import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.inference.mcmc.sample_results import SampleResult
from bayes_vi.inference.mcmc.stepsize_adaptation_kernels import StepSizeAdaptationKernel
from bayes_vi.inference.mcmc.transition_kernels import TransitionKernel, HamiltonianMonteCarlo
from bayes_vi.models import Model
from bayes_vi.utils import compose
from bayes_vi.utils import to_ordered_dict, make_transformed_log_prob

tfd = tfp.distributions
tfb = tfp.bijectors


class MCMC(Inference):

    def __init__(self,
                 model: Model,
                 dataset: tf.data.Dataset,
                 transition_kernel: TransitionKernel = HamiltonianMonteCarlo(step_size=0.01, num_leapfrog_steps=50),
                 step_size_adaptation_kernel: StepSizeAdaptationKernel = None,
                 transforming_bijector: tfb.Bijector = None):
        super(MCMC, self).__init__(model=model, dataset=dataset)
        # take num_datapoints as a single batch and extract features and targets
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]

        # condition model on features
        self.model = model(features=self.features)

        # define target_log_prob_fn as the log probability of the model, being the unnormalized log posterior
        target_log_prob = functools.partial(self.model.unnormalized_log_posterior, targets=self.targets)
        self.target_log_prob_fn = lambda *state: target_log_prob(
            to_ordered_dict(self.model.param_names, state)
        )

        # set given transition kernel and transforming bijector
        self.transition_kernel = transition_kernel
        self.step_size_adaptation_kernel = step_size_adaptation_kernel
        self.transforming_bijector = transforming_bijector

    def fit(self,
            initial_state: 'collections.OrderedDict[str, tf.Tensor]' = None,
            jitter: float = 0.1,
            num_chains: int = 4,
            num_samples: int = 4000,
            num_burnin_steps: int = 1000,
            merge_state_parts: bool = False):

        # generate initial state for markov chain
        if initial_state is None:
            # sample initial state from priors
            initial_state = self.model.unconstrain_state(
                list(self.model.prior_distribution.sample(num_chains).values())
            )
        elif initial_state == 'ones':
            # set initial state to all ones
            sample = self.model.unconstrain_state(
                [tf.ones_like(x) for x in self.model.prior_distribution.sample().values()]
            )
            initial_state = [
                tf.random.normal([num_chains] + part.shape, mean=part, stddev=min(0.0001, max(1., float(jitter))))
                for part in sample
            ]
        elif isinstance(initial_state, collections.OrderedDict):
            # define initial state based on given OrderedDict with explicit values for each parameter
            initial_state = self.model.unconstrain_state(
                list(initial_state.values())
            )
        else:
            raise ValueError("invalid initial_state, has to be in [None, 'ones', collections.OrderedDict]")

        # components of `initial_state` are considered independent
        # if a transforming_bijector is given or merge_states in explicitly set,
        # we construct a single component initial state, such that all parameters are considered dependent
        if self.transforming_bijector or merge_state_parts:
            initial_state = [self.model.split_bijector.inverse(initial_state)]
            split_bijector = self.model.split_bijector
        else:
            split_bijector = None

        transformed_log_prob = make_transformed_log_prob(self.target_log_prob_fn,
                                                         bijector=self.model.reshape_constrain_bijectors,
                                                         direction='forward',
                                                         targets_fixed=True,
                                                         split_bijector=split_bijector)

        # build given Markov transition kernel
        kernel = self.transition_kernel(transformed_log_prob)
        # add transition kernels trace_fn to list of trace functions (for later composition)
        trace_fns = [self.transition_kernel.trace_fn]

        if self.transforming_bijector:
            # wrap `TransitionKernel` in a `TransformedTransitionKernel`,
            kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=kernel,
                bijector=self.transforming_bijector
            )
            # append another trace function reaching through to the inner kernel
            trace_fns.append(lambda _, pkr: (_, pkr.inner_results))


        # wrap `TransitionKernel` in a `StepSizeAdaptationKernel` if one is provided
        if self.step_size_adaptation_kernel:
            kernel = self.step_size_adaptation_kernel(inner_kernel=kernel)
            # append another trace function reaching through to the inner kernel
            trace_fns.append(self.step_size_adaptation_kernel.trace_fn)

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
        if self.transforming_bijector or merge_state_parts:
            samples = self.model.transform_state_forward(samples[0], to_dict=False)
        else:
            samples = self.model.transform_state_forward(samples, split=False, to_dict=False)

        # reduce one dimensional params to scalar
        samples = [tf.reshape(s, shape=s.shape[:-1]) if s.shape[-1] == 1 else s for s in samples]
        # return `SampleResult` object containing samples, trace and statistics
        return SampleResult(self.model, samples, trace)

    @staticmethod
    @tf.function
    def sample_chain(num_results, num_burnin_steps, current_state, kernel, trace_fn):
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=current_state,
            kernel=kernel,
            trace_fn=trace_fn
        )
