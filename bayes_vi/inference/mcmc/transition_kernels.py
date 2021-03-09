"""Module providing different Markov transition kernels."""

import functools

import tensorflow_probability as tfp

from bayes_vi.inference.mcmc.stepsize_adaptation_kernels import StepSizeAdaptationKernel

tfb = tfp.bijectors


class TransitionKernel:
    """Base class for Markov transition Kernels."""

    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    def __call__(self, target_log_prob_fn):
        kernel = self.kernel(target_log_prob_fn)
        trace_fns = [self.trace_fn]
        if hasattr(self, 'stepsize_adaptation_kernel') \
                and isinstance(self.stepsize_adaptation_kernel, StepSizeAdaptationKernel):
            kernel = self.stepsize_adaptation_kernel(kernel)
            trace_fns.append(lambda _, pkr: (_, pkr.inner_results))
        if hasattr(self, 'transforming_bijector') \
                and isinstance(self.transforming_bijector, tfb.Bijector):
            kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=kernel,
                bijector=self.transforming_bijector
            )
            trace_fns.append(lambda _, pkr: (_, pkr.inner_results))
        return kernel, trace_fns, self.trace_fn_metrics


class HamiltonianMonteCarlo(TransitionKernel):
    """Implements HMC transition kernel.

    Note: This is a wrapper around `tfp.mcmc.HamiltonianMonteCarlo`.

    Attributes
    ----------
    step_size: `float` or `list` of `float`
        representing the step size for the leapfrog integrator.
        Must broadcast with the shape of the state.
        Larger step sizes lead to faster progress, but too-large step sizes make
        rejection exponentially more likely. When possible, it's often helpful
        to match per-variable step sizes to the standard deviations of the
        target distribution in each variable.
    num_leapfrog_steps: `int`
        Number of steps to run the leapfrog integrator for.
        Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
    kernel: `callable`
        A callable taking a target_log_prob_fn and returning
        `tfp.mcmc.HamiltonianMonteCarlo` with the specified parameters.
    step_size_adaptation_kernel: `bayes_vi.inference.mcmc.stepsize_adaptation_kernels.StepSizeAdaptationKernel`
        A stepsize adaptation kernel to wrap the transition kernel and optimize stepsize in burnin phase.
        (Default: `None`)
    transforming_bijectors: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
        A single or per state part transforming bijector to transform the generated samples.
        This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
        the target distribution for more efficient sampling.
        In the context of HMC this is approximately Riemannian-HMC (RHMC).
    state_gradients_are_stopped: `bool`
        A boolean indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        (Default: `False` (i.e., do not apply `stop_gradient`)).
    name: `str`
        Name prefixed to Ops created by this function.
        (Default: `None` (i.e., 'hmc_kernel')).
    """

    def __init__(self, step_size=0.01, num_leapfrog_steps=5,
                 transforming_bijector=None, stepsize_adaptation_kernel=None,
                 state_gradients_are_stopped=False, name=None):
        """Initializes the HMC kernel.

        Note: This is a wrapper around `tfp.mcmc.HamiltonianMonteCarlo`

        Parameters
        ----------
        step_size: `float` or `list` of `float`
            representing the step size for the leapfrog integrator.
            Must broadcast with the shape of the state.
            Larger step sizes lead to faster progress, but too-large step sizes make
            rejection exponentially more likely. When possible, it's often helpful
            to match per-variable step sizes to the standard deviations of the
            target distribution in each variable. (Default: `0.01`)
        num_leapfrog_steps: `int`
            Number of steps to run the leapfrog integrator for.
            Total progress per HMC step is roughly proportional to
            `step_size * num_leapfrog_steps`. (Default: `5`)
        step_size_adaptation_kernel: `bayes_vi.inference.mcmc.stepsize_adaptation_kernels.StepSizeAdaptationKernel`
            A stepsize adaptation kernel to wrap the transition kernel and optimize stepsize in burnin phase.
            (Default: `None`)
        transforming_bijector: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
            A single or per state part transforming bijector to transform the generated samples.
            This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
            the target distribution for more efficient sampling.
            In the context of HMC this is corresponds to Riemannian-HMC (RHMC).
        state_gradients_are_stopped: `bool`
            A boolean indicating that the proposed
            new state be run through `tf.stop_gradient`. This is particularly useful
            when combining optimization over samples from the HMC chain.
            (Default: `False` (i.e., do not apply `stop_gradient`)).
        name: `str`
            Name prefixed to Ops created by this function.
            (Default: `None` (i.e., 'hmc_kernel'))."""
        super(HamiltonianMonteCarlo, self).__init__(name)
        self.transforming_bijector = transforming_bijector
        self.stepsize_adaptation_kernel = stepsize_adaptation_kernel
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.state_gradients_are_stopped = state_gradients_are_stopped
        self.kernel = functools.partial(
            tfp.mcmc.HamiltonianMonteCarlo,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            name=name
        )
        self.trace_fn_metrics = ['is_accepted', 'target_log_prob', 'step_size']

    @staticmethod
    def trace_fn(_, pkr):
        return (
            pkr.is_accepted,
            pkr.accepted_results.target_log_prob,
            pkr.accepted_results.step_size
        )


class NoUTurnSampler(TransitionKernel):
    """Implementation of the NoUTurnSampler.

    Note: This is a wrapper around `tfp.mcmc.NoUTurnSampler`.

    Attributes
    ----------
    step_size: `float` or `list` of `float`
        representing the step size for the leapfrog integrator.
        Must broadcast with the shape of the state.
        Larger step sizes lead to faster progress, but too-large step sizes make
        rejection exponentially more likely. When possible, it's often helpful
        to match per-variable step sizes to the standard deviations of the
        target distribution in each variable.
    max_tree_depth: `int`
        Maximum depth of the tree implicitly built by NUTS. The
        maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
        the number of nodes in a binary tree `max_tree_depth` nodes deep. The
        default setting of 10 takes up to 1024 leapfrog steps. (Default: `10`)
    max_energy_diff: `float`
        Threshold of energy differences at each leapfrog,
        divergence samples are defined as leapfrog steps that exceed this
        threshold. (Default: `1000`).
    unrolled_leapfrog_steps: `int`
        Number of leapfrogs to unroll per tree expansion step.
        Applies a direct linear multiplier to the maximum
        trajectory length implied by max_tree_depth. (Default: `1`).
    parallel_iterations: `int`
        Number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
        (Default: `10`).
    kernel: `callable`
        A callable taking a target_log_prob_fn and returning
        `tfp.mcmc.NoUTurnSampler` with the specified parameters.
    step_size_adaptation_kernel: `bayes_vi.inference.mcmc.stepsize_adaptation_kernels.StepSizeAdaptationKernel`
        A stepsize adaptation kernel to wrap the transition kernel and optimize stepsize in burnin phase.
        (Default: `None`)
    transforming_bijector: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
        A single or per state part transforming bijector to transform the generated samples.
        This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
        the target distribution for more efficient sampling.
        In the context of HMC this is approximately Riemannian-HMC (RHMC).
    name: `str`
        Name prefixed to Ops created by this function.
        (Default: `None` (i.e., 'nuts_kernel')).
    """

    def __init__(self, step_size=0.01, max_tree_depth=5, max_energy_diff=1000.0,
                 unrolled_leapfrog_steps=1, parallel_iterations=10,
                 transforming_bijector=None, stepsize_adaptation_kernel=None, name=None):
        """Initializes the NoUTurnSampler kernel.

        Parameters
        ----------
        step_size: `float` or `list` of `float`
            Representing the step size for the leapfrog integrator.
            Must broadcast with the shape of the state.
            Larger step sizes lead to faster progress, but too-large step sizes make
            rejection exponentially more likely. When possible, it's often helpful
            to match per-variable step sizes to the standard deviations of the
            target distribution in each variable. (Default: `0.01`)
        max_tree_depth: `int`
            Maximum depth of the tree implicitly built by NUTS. The
            maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
            the number of nodes in a binary tree `max_tree_depth` nodes deep. The
            default setting of 10 takes up to 1024 leapfrog steps. (Default: `5`)
        max_energy_diff: `float`
            Threshold of energy differences at each leapfrog,
            divergence samples are defined as leapfrog steps that exceed this
            threshold. (Default: `1000`).
        unrolled_leapfrog_steps: `int`
            Number of leapfrogs to unroll per tree expansion step.
            Applies a direct linear multiplier to the maximum
            trajectory length implied by max_tree_depth. (Default: `1`).
        parallel_iterations: `int`
            Number of iterations allowed to run in parallel.
            It must be a positive integer. See `tf.while_loop` for more details.
            (Default: `10`).
        step_size_adaptation_kernel: `bayes_vi.inference.mcmc.stepsize_adaptation_kernels.StepSizeAdaptationKernel`
            A stepsize adaptation kernel to wrap the transition kernel and optimize stepsize in burnin phase.
            (Default: `None`)
        transforming_bijectors: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
            A single or per state part transforming bijector to transform the generated samples.
            This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
            the target distribution for more efficient sampling.
            In the context of HMC this is approximately Riemannian-HMC (RHMC).
        name: `str`
            Name prefixed to Ops created by this function.
            (Default: `None` (i.e., 'nuts_kernel')).
        """
        super(NoUTurnSampler, self).__init__(name)
        self.transforming_bijector = transforming_bijector
        self.stepsize_adaptation_kernel = stepsize_adaptation_kernel
        self.step_size = step_size
        self.max_tree_depth = max_tree_depth
        self.max_energy_diff = max_energy_diff
        self.unrolled_leapfrog_steps = unrolled_leapfrog_steps
        self.parallel_iterations = parallel_iterations
        self.kernel = functools.partial(
            tfp.mcmc.NoUTurnSampler,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            max_energy_diff=max_energy_diff,
            unrolled_leapfrog_steps=unrolled_leapfrog_steps,
            parallel_iterations=parallel_iterations,
            name=name
        )
        self.trace_fn_metrics = [
            'is_accepted', 'target_log_prob', 'log_accept_ratio', 'step_size',
            'leapfrogs_taken', 'has_divergence', 'energy'
        ]

    @staticmethod
    def trace_fn(_, pkr):
        return (
            pkr.is_accepted,
            pkr.target_log_prob,
            pkr.log_accept_ratio,
            pkr.step_size,
            pkr.leapfrogs_taken,
            pkr.has_divergence,
            pkr.energy
        )


class RandomWalkMetropolis(TransitionKernel):
    """Implementation of the RandomWalkMetropolis kernel.

    Note: This is a wrapper around `tfp.mcmc.RandomWalkMetropolis`

    Attributes
    ----------
    transforming_bijectors: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
        A single or per state part transforming bijector to transform the generated samples.
        This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
        the target distribution for more efficient sampling.
        In the context of HMC this is approximately Riemannian-HMC (RHMC).
    new_state_fn: `callable`
        Callable which takes a list of state parts and a seed;
        returns a same-type `list` of `Tensor`s, each being a perturbation
        of the input state parts. The perturbation distribution is assumed to be
        a symmetric distribution centered at the input state part.
        (Default: `None` which is mapped to `tfp.mcmc.random_walk_normal_fn()`).
    kernel: `callable`
        A callable taking a target_log_prob_fn and returning
        `tfp.mcmc.RandomWalkMetropolis` with the specified parameters.
    transforming_bijectors: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
        A single or per state part transforming bijector to transform the generated samples.
        This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
        the target distribution for more efficient sampling.
    name: `str`
        Name prefixed to Ops created by this function.
        (Default: `None` (i.e., 'rwm_kernel')).
    """

    def __init__(self, transforming_bijector=None, new_state_fn=None, name=None):
        """Initializes the RandomWalkMetropolis kernel.

        Attributes
        ----------
        transforming_bijector: `tfp.bijectors.Bijector` or `list` of `tfp.bijectors.Bijector`
            A single or per state part transforming bijector to transform the generated samples.
            This allows trainable bijectors to be applied to achieve decorrelation between parameters and simplifying
            the target distribution for more efficient sampling.
            In the context of HMC this is approximately Riemannian-HMC (RHMC).
        new_state_fn: `callable`
            Callable which takes a list of state parts and a seed;
            returns a same-type `list` of `Tensor`s, each being a perturbation
            of the input state parts. The perturbation distribution is assumed to be
            a symmetric distribution centered at the input state part.
            (Default: `None` which is mapped to `tfp.mcmc.random_walk_normal_fn()`).
        name: `str`
            Name prefixed to Ops created by this function.
            (Default: `None` (i.e., 'rwm_kernel')).

        """
        super(RandomWalkMetropolis, self).__init__(name)
        self.transforming_bijector = transforming_bijector
        self.new_state_fn = new_state_fn
        self.kernel = functools.partial(tfp.mcmc.RandomWalkMetropolis,
                                        new_state_fn=new_state_fn,
                                        name=name)
        self.trace_fn_metrics = ['is_accepted']

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)
