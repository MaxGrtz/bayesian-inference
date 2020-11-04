import collections
import functools
from typing import Callable, Any, List

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.mcmc.sample_results import SampleResult
from bayes_vi.mcmc.stepsize_adaptation_kernels import StepSizeAdaptationKernel
from bayes_vi.mcmc.transition_kernels import TransitionKernel, HamiltonianMonteCarlo
from bayes_vi.models import BayesianModel
from bayes_vi.models import Model
from bayes_vi.utils.bijectors import TransformReshapeSplit
from bayes_vi.utils.functions import compose
from bayes_vi.vi.surrogate_posterior import SurrogatePosterior, MeanFieldADVI

tfd = tfp.distributions
tfb = tfp.bijectors


class Inference:

    def __init__(self, model: Model, dataset: tf.data.Dataset):
        self.model = model
        self.dataset = dataset


class MCMC(Inference):

    def __init__(self,
                 model: BayesianModel,
                 dataset: tf.data.Dataset,
                 transition_kernel: TransitionKernel = HamiltonianMonteCarlo(step_size=0.01, num_leapfrog_steps=50),
                 transforming_bijector: tfb.Bijector = None,
                 step_size_adaptation_kernel: StepSizeAdaptationKernel = None):
        super(MCMC, self).__init__(model=model, dataset=dataset)
        # take num_datapoints as a single batch and extract features and targets
        batch = list(dataset.batch(dataset.cardinality()).take(1))[0]
        try:
            self.features, self.targets = batch
        except ValueError:
            self.features, self.targets = None, batch

        # initialize model with features and targets
        self.model = model(features=self.features, targets=self.targets)

        # define target_log_prob_fn as the log probability of the model, being the unnormalized log posterior
        self.target_log_prob_fn = lambda *args: self.model.joint_distribution.log_prob(
            dict(
                **dict(zip(self.model.param_names, args)),
                y=self.targets
            )
        )

        # set given transition kernel and transforming bijector
        self.transition_kernel = transition_kernel
        self.step_size_adaptation_kernel = step_size_adaptation_kernel
        self.transforming_bijector = transforming_bijector

    def sample(self,
               num_chains: int = 4,
               num_samples: int = 4000,
               num_burnin_steps: int = 1000,
               initial_sample: 'collections.OrderedDict[str, tf.Tensor]' = None,
               jitter: float = 0.1,
               merge_states: bool = False):

        # generate initial state for markov chain
        if initial_sample is None:
            # sample initial state from priors
            initial_state = list(self.model.joint_distribution.sample(num_chains).values())[:-1]
        elif initial_sample == 'ones':
            # set initial state to all ones
            initial_state = tf.nest.map_structure(
                lambda x: tf.random.normal(shape=(1,), mean=tf.ones_like(x), stddev=jitter),
                list(self.model.joint_distribution.sample(num_chains).values())[:-1]
            )
        elif isinstance(initial_sample, collections.OrderedDict):
            # define initial state based on given OrderedDict with explicit values for each parameter
            initial_state = list(initial_sample.values())
        else:
            raise ValueError("invalid initial_sample, has to be in [None, 'ones', collections.OrderedDict]")


        # build given Markov transition kernel (called transformed_kernel for convenience)
        transformed_kernel = self.transition_kernel(self.target_log_prob_fn)
        # add transition kernels trace_fn to list of trace functions (for later composition)
        trace_fns = [self.transition_kernel.trace_fn]

        # construct transformation `tfp.bijectors.Bijector`
        # components of `initial_state` are considered independent
        # if a transforming_bijector is given or merge_states in explicitly set,
        # we construct a single component initial state, such that all parameters are considered dependent
        if self.transforming_bijector or merge_states:
            bijectors = []
            # figure out the size of (number of elements in) each component
            block_sizes = [tf.size(part[0]) for part in initial_state]
            # construct `tfp.bijectors.Split` from those `block_sizes` for splitting/merging of components
            split_bijector = tfb.Split(num_or_size_splits=block_sizes)

            reshaping_bijectors = [tfb.Reshape(event_shape_in=(part.shape[0], -1), event_shape_out=part.shape)
                                   for part in initial_state]

            # add transforming bijectors, where `constraining_bijectors`
            # project all parameter values into the correctly constrained part of R^n
            # and `transforming_bijector` allows for learning local metric/ decorrelation
            bijectors.append(TransformReshapeSplit(
                transforming_bijectors=self.model.constraining_bijectors,
                reshaping_bijectors=reshaping_bijectors,
                split_bijector=split_bijector))

            bijectors.append(split_bijector)
            if self.transforming_bijector:
                bijectors.append(self.transforming_bijector)

            # chain the transforming bijectors
            bijector = tfb.Chain(bijectors)
        else:
            # leave components of `initial_state_unconstrained` independent
            # and simply provide constraining bijector for each component with `constraining_bijectors`
            bijector = self.model.constraining_bijectors

        # wrap `TransitionKernel` in a `TransformedTransitionKernel`,
        # applying all the above defined transformations via a chained `tfp.bijectors.Bijector`
        transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=transformed_kernel,
            bijector=bijector
        )
        # append another trace function reaching through to the inner kernel
        trace_fns.append(lambda _, pkr: (_, pkr.inner_results))

        # wrap `TransitionKernel` in a `StepSizeAdaptationKernel` if one is given
        if self.step_size_adaptation_kernel:
            transformed_kernel = self.step_size_adaptation_kernel(inner_kernel=transformed_kernel)
            # append another trace function reaching through to the inner kernel
            trace_fns.append(self.step_size_adaptation_kernel.trace_fn)

        # define `trace_fn` for Markov chain as composition of all defined trace functions
        trace_fn = compose(list(reversed(trace_fns)))

        # apply `tfp.mcmc.sample_chain` with a convenience wrapper
        # to allow for annotation with `tf.function` giving performance boost
        results = self.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=transformed_kernel,
            trace_fn=trace_fn
        )
        # return `SampleResult` object containing samples, trace and statistics
        return SampleResult(self.model, *results)

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


class VI(Inference):

    def __init__(self,
                 model: BayesianModel,
                 dataset: tf.data.Dataset,
                 surrogate_posterior: SurrogatePosterior = MeanFieldADVI(),
                 divergence_fn: Callable[[Any], Any] = tfp.vi.kl_reverse):
        super(VI, self).__init__(model=model, dataset=dataset)
        try:
            self.features, self.targets = list(dataset.batch(dataset.cardinality()).take(1))[0]
        except ValueError:
            self.features, self.targets = None, list(dataset.batch(dataset.cardinality()).take(1))[0]
        self.model = model(features=self.features, targets=self.targets)
        self.target_log_prob_fn = lambda *args: self.model.joint_distribution.log_prob(
            dict(**dict(zip(self.model.param_names, args)), y=self.targets))
        self.surrogate_posterior_build = surrogate_posterior(self.model, self.model.constraining_bijectors)
        self.loss = functools.partial(tfp.vi.monte_carlo_variational_loss,
                                      discrepancy_fn=divergence_fn,
                                      use_reparameterization=True)

    def fit(self,
            optimizer: tf.optimizers.Optimizer = tf.optimizers.Adam(),
            num_steps: int = 10000,
            sample_size: int = 1,
            convergence_criterion: tfp.optimizer.convergence_criteria.ConvergenceCriterion = None,
            trainable_variables: List[Any] = None,
            seed: Any = None,
            name: str = 'fit_surrogate_posterior'):
        return self.surrogate_posterior_build, tfp.vi.fit_surrogate_posterior(
            self.target_log_prob_fn, self.surrogate_posterior_build, optimizer, num_steps,
            convergence_criterion=convergence_criterion, variational_loss_fn=self.loss,
            sample_size=sample_size, trainable_variables=trainable_variables, seed=seed, name=name
        )


class MLE(Inference):

    def __init__(self, model: Model, dataset: tf.data.Dataset):
        super(MLE, self).__init__(model=model, dataset=dataset)
        self.model = model
        self.dataset = dataset
