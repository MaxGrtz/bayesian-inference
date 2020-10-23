import tensorflow_probability as tfp
import functools


class StepSizeAdaptationKernel:

    def __init__(self, num_adaptation_steps, target_accept_prob, step_size_setter_fn,
                 step_size_getter_fn, log_accept_prob_getter_fn):
        self.num_adaptation_steps = num_adaptation_steps
        self.target_accept_prob = target_accept_prob
        self.step_size_setter_fn = step_size_setter_fn
        self.step_size_getter_fn = step_size_getter_fn
        self.log_accept_prob_getter_fn = log_accept_prob_getter_fn

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Not yet Implemented')


class SimpleStepSizeAdaptation(StepSizeAdaptationKernel):

    def __init__(self, num_adaptation_steps, target_accept_prob=0.75, adaptation_rate=0.01,
                 step_size_setter_fn=None, step_size_getter_fn=None, log_accept_prob_getter_fn=None):
        super(SimpleStepSizeAdaptation, self).__init__(num_adaptation_steps, target_accept_prob,
                                                       step_size_setter_fn, step_size_getter_fn,
                                                       log_accept_prob_getter_fn)
        self.adaptation_rate = adaptation_rate
        self.kernel = functools.partial(tfp.mcmc.SimpleStepSizeAdaptation,
                                        num_adaptation_steps=num_adaptation_steps,
                                        target_accept_prob=target_accept_prob,
                                        adaptation_rate=adaptation_rate)

    def __call__(self, inner_kernel):
        return self.kernel(inner_kernel)

    @staticmethod
    def trace_fn(_, pkr):
        return _, pkr.inner_results


class DualAveragingStepSizeAdaptation(StepSizeAdaptationKernel):

    def __init__(self, num_adaptation_steps, target_accept_prob=0.75, exploration_shrinkage=0.05,
                 shrinkage_target=None, step_count_smoothing=10, decay_rate=0.75,
                 step_size_setter_fn=None, step_size_getter_fn=None, log_accept_prob_getter_fn=None):
        super(DualAveragingStepSizeAdaptation, self).__init__(num_adaptation_steps, target_accept_prob,
                                                              step_size_setter_fn, step_size_getter_fn,
                                                              log_accept_prob_getter_fn)
        self.exploration_shrinkage = exploration_shrinkage
        self.shrinkage_target = shrinkage_target
        self.step_count_smoothing = step_count_smoothing
        self.decay_rate = decay_rate

        self.kernel = functools.partial(tfp.mcmc.DualAveragingStepSizeAdaptation,
                                        num_adaptation_steps=num_adaptation_steps,
                                        target_accept_prob=target_accept_prob,
                                        exploration_shrinkage=exploration_shrinkage,
                                        shrinkage_target=shrinkage_target,
                                        step_count_smoothing=step_count_smoothing,
                                        decay_rate=decay_rate)

    def __call__(self, inner_kernel):
        return self.kernel(inner_kernel)

    @staticmethod
    def trace_fn(_, pkr):
        return _, pkr.inner_results
