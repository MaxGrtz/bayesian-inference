import collections
import functools

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.inference import Inference
from bayes_vi.inference.mcmc.sample_results import SampleResult
from bayes_vi.inference.mcmc.stepsize_adaptation_kernels import StepSizeAdaptationKernel
from bayes_vi.inference.mcmc.transition_kernels import TransitionKernel, RandomWalkMetropolis
from bayes_vi.models import Model
from bayes_vi.utils import compose
from bayes_vi.utils import to_ordered_dict

tfd = tfp.distributions
tfb = tfp.bijectors


class MCMC(Inference):

    def __init__(self,
                 model: Model,
                 dataset: tf.data.Dataset,
                 transition_kernel: TransitionKernel = RandomWalkMetropolis(),
                 step_size_adaptation_kernel: StepSizeAdaptationKernel = None,
                 transforming_bijectors: tfb.Bijector = None):
        super(MCMC, self).__init__(model=model, dataset=dataset)
        # take num_datapoints as a single batch and extract features and targets
        self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]

        # condition model on features
        self.model = model(features=self.features)

        # define target_log_prob_fn as the log probability of the model, being the unnormalized log posterior
        self.target_log_prob = functools.partial(self.model.unnormalized_log_posterior, targets=self.targets)

        # set given transition kernel and transforming bijector
        self.transition_kernel = transition_kernel
        self.step_size_adaptation_kernel = step_size_adaptation_kernel
        self.transforming_bijectors = transforming_bijectors

    def fit(self,
            initial_state: 'collections.OrderedDict[str, tf.Tensor]' = None,
            jitter: float = 0.001,
            num_chains: int = 4,
            num_samples: int = 4000,
            num_burnin_steps: int = 1000,
            merge_state_parts: bool = False):

        # generate initial state for markov chain
        if initial_state is None:
            # sample initial state from priors
            initial_state = list(self.model.prior_distribution.sample(num_chains).values())
        elif initial_state == 'ones':
            # set initial state to all ones
            sample = [tf.ones_like(x) for x in self.model.prior_distribution.sample().values()]
            initial_state = [
                tf.random.normal([num_chains] + part.shape, mean=part, stddev=max(1., float(jitter)))
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
            samples = [samples]

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
