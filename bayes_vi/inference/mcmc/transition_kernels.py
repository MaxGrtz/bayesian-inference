import functools

import tensorflow_probability as tfp


class TransitionKernel:

    def __init__(self, seed=None, name=None):
        self.seed = seed
        self.name = name

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Not yet Implemented')


class HamiltonianMonteCarlo(TransitionKernel):

    def __init__(self,
                 step_size,
                 num_leapfrog_steps,
                 state_gradients_are_stopped=False,
                 seed=None,
                 name=None):
        super(HamiltonianMonteCarlo, self).__init__(seed, name)
        self.state_gradients_are_stopped = state_gradients_are_stopped
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.kernel = functools.partial(tfp.mcmc.HamiltonianMonteCarlo, step_size=step_size,
                                        num_leapfrog_steps=num_leapfrog_steps,
                                        state_gradients_are_stopped=state_gradients_are_stopped,
                                        seed=seed, name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)


class NoUTurnSampler(TransitionKernel):

    def __init__(self,
                 step_size,
                 max_tree_depth=10,
                 max_energy_diff=1000.0,
                 unrolled_leapfrog_steps=1,
                 parallel_iterations=10,
                 seed=None,
                 name=None):
        super(NoUTurnSampler, self).__init__(seed, name)
        self.step_size = step_size
        self.max_tree_depth = max_tree_depth
        self.max_energy_diff = max_energy_diff
        self.unrolled_leapfrog_steps = unrolled_leapfrog_steps
        self.parallel_iterations = parallel_iterations
        self.kernel = functools.partial(tfp.mcmc.NoUTurnSampler, step_size=step_size,
                                        max_tree_depth=max_tree_depth,
                                        max_energy_diff=max_energy_diff,
                                        unrolled_leapfrog_steps=unrolled_leapfrog_steps,
                                        parallel_iterations=parallel_iterations, seed=seed, name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)


class RandomWalkMetropolis(TransitionKernel):

    def __init__(self, new_state_fn=None, seed=None, name=None):
        super(RandomWalkMetropolis, self).__init__(seed, name)
        self.new_state_fn = new_state_fn
        self.kernel = functools.partial(tfp.mcmc.RandomWalkMetropolis, new_state_fn=new_state_fn,
                                        seed=seed, name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)


class MetropolisAdjustedLangevinAlgorithm(TransitionKernel):

    def __init__(self,
                 step_size,
                 volatility_fn=None,
                 parallel_iterations=10,
                 seed=None,
                 name=None):
        super(MetropolisAdjustedLangevinAlgorithm, self).__init__(seed, name)
        self.step_size = step_size
        self.volatility_fn = volatility_fn
        self.parallel_iterations = parallel_iterations
        self.kernel = functools.partial(tfp.mcmc.MetropolisAdjustedLangevinAlgorithm,
                                        step_size=step_size,
                                        volatility_fn=volatility_fn,
                                        parallel_iterations=parallel_iterations,
                                        seed=seed,
                                        name=name)

    def __call__(self, target_log_prob_fn):
        return self.kernel(target_log_prob_fn)

    @staticmethod
    def trace_fn(_, pkr):
        return (pkr.is_accepted,)
