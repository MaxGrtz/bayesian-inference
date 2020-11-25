import functools

import tensorflow_probability as tfp


class TransitionKernel:
    """Base class for Markov transition Kernels."""

    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Not yet Implemented')


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
    state_gradients_are_stopped: `bool`
        A boolean indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        (Default: `False` (i.e., do not apply `stop_gradient`)).
    name: `str`
        Name prefixed to Ops created by this function.
        (Default: `None` (i.e., 'hmc_kernel')).
    """

    def __init__(self, step_size, num_leapfrog_steps, state_gradients_are_stopped=False, name=None):
        """Initializes the HMC kernel.

        Parameters
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
        state_gradients_are_stopped: `bool`
            A boolean indicating that the proposed
            new state be run through `tf.stop_gradient`. This is particularly useful
            when combining optimization over samples from the HMC chain.
            (Default: `False` (i.e., do not apply `stop_gradient`)).
        name: `str`
            Name prefixed to Ops created by this function.
            (Default: `None` (i.e., 'hmc_kernel'))."""
        super(HamiltonianMonteCarlo, self).__init__(name)
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.state_gradients_are_stopped = state_gradients_are_stopped
        self.kernel = functools.partial(tfp.mcmc.HamiltonianMonteCarlo,
                                        step_size=step_size,
                                        num_leapfrog_steps=num_leapfrog_steps,
                                        state_gradients_are_stopped=state_gradients_are_stopped,
                                        name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)


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
    name: `str`
        Name prefixed to Ops created by this function.
        (Default: `None` (i.e., 'nuts_kernel')).
    """

    def __init__(self, step_size, max_tree_depth=10, max_energy_diff=1000.0,
                 unrolled_leapfrog_steps=1, parallel_iterations=10, name=None):
        """Initializes the NoUTurnSampler kernel.

        Parameters
        ----------
        step_size: `float` or `list` of `float`
            Representing the step size for the leapfrog integrator.
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
        name: `str`
            Name prefixed to Ops created by this function.
            (Default: `None` (i.e., 'nuts_kernel')).
        """
        super(NoUTurnSampler, self).__init__(name)
        self.step_size = step_size
        self.max_tree_depth = max_tree_depth
        self.max_energy_diff = max_energy_diff
        self.unrolled_leapfrog_steps = unrolled_leapfrog_steps
        self.parallel_iterations = parallel_iterations
        self.kernel = functools.partial(tfp.mcmc.NoUTurnSampler,
                                        step_size=step_size,
                                        max_tree_depth=max_tree_depth,
                                        max_energy_diff=max_energy_diff,
                                        unrolled_leapfrog_steps=unrolled_leapfrog_steps,
                                        parallel_iterations=parallel_iterations,
                                        name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)


class RandomWalkMetropolis(TransitionKernel):
    """Implementation of the RandomWalkMetropolis kernel.

    Note: This is a wrapper around `tfp.mcmc.RandomWalkMetropolis`

    Attributes
    ----------
    new_state_fn: `callable`
        Callable which takes a list of state parts and a seed;
        returns a same-type `list` of `Tensor`s, each being a perturbation
        of the input state parts. The perturbation distribution is assumed to be
        a symmetric distribution centered at the input state part.
        (Default: `None` which is mapped to `tfp.mcmc.random_walk_normal_fn()`).
    kernel: `callable`
        A callable taking a target_log_prob_fn and returning
        `tfp.mcmc.RandomWalkMetropolis` with the specified parameters.
    name: `str`
        Name prefixed to Ops created by this function.
        (Default: `None` (i.e., 'rwm_kernel')).
    """

    def __init__(self, new_state_fn=None, name=None):
        """Initializes the RandomWalkMetropolis kernel.

        Attributes
        ----------
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
        self.new_state_fn = new_state_fn
        self.kernel = functools.partial(tfp.mcmc.RandomWalkMetropolis,
                                        new_state_fn=new_state_fn,
                                        name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)


class MetropolisAdjustedLangevinAlgorithm(TransitionKernel):
    """Implementation of the MetropolisAdjustedLangevinAlgorithm.

    Note: This is a wrapper around `tfp.mcmc.MetropolisAdjustedLangevinAlgorithm`

    Attributes
    ----------
    step_size: `float` or `list` of `float`
        Representing the step size for the leapfrog integrator.
        Must broadcast with the shape of current state.
        Larger step sizes lead to faster progress, but too-large step sizes
        make rejection exponentially more likely. When possible,
        it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
    volatility_fn: `callable`
        Callable which takes an argument like current state
        (or `*current_state` if it's a list) and returns
        volatility value at current state. Should return a `Tensor` or Python
        `list` of `Tensor`s that must broadcast with the shape of
        current state Defaults to the identity function.
        (Default: `None`).
    parallel_iterations: `int`
        Number of coordinates for which the gradients of
        the volatility matrix `volatility_fn` can be computed in parallel.
        (Default: `10`).
    kernel: `callable`
        A callable taking a target_log_prob_fn and returning
        `tfp.mcmc.MetropolisAdjustedLangevinAlgorithm` with the specified parameters.
    name: `str`
        Name prefixed to Ops created by this function.
        (Default: `None` (i.e., 'mala_kernel')).
    """

    def __init__(self, step_size, volatility_fn=None, parallel_iterations=10, name=None):
        """Initializes the MetropolisAdjustedLangevinAlgorithm.

        Parameters
        ----------
        step_size: `float` or `list` of `float`
            Representing the step size for the leapfrog integrator.
            Must broadcast with the shape of current state.
            Larger step sizes lead to faster progress, but too-large step sizes
            make rejection exponentially more likely. When possible,
            it's often helpful to match per-variable step sizes to the
            standard deviations of the target distribution in each variable.
        volatility_fn: `callable`
            Callable which takes an argument like current state
            (or `*current_state` if it's a list) and returns
            volatility value at current state. Should return a `Tensor` or Python
            `list` of `Tensor`s that must broadcast with the shape of
            current state Defaults to the identity function.
            (Default: `None`).
        parallel_iterations: `int`
            Number of coordinates for which the gradients of
            the volatility matrix `volatility_fn` can be computed in parallel.
            (Default: `10`).
        name: `str`
            Name prefixed to Ops created by this function.
            (Default: `None` (i.e., 'mala_kernel')).

        """
        super(MetropolisAdjustedLangevinAlgorithm, self).__init__(name)
        self.step_size = step_size
        self.volatility_fn = volatility_fn
        self.parallel_iterations = parallel_iterations
        self.kernel = functools.partial(tfp.mcmc.MetropolisAdjustedLangevinAlgorithm,
                                        step_size=step_size,
                                        volatility_fn=volatility_fn,
                                        parallel_iterations=parallel_iterations,
                                        name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)
