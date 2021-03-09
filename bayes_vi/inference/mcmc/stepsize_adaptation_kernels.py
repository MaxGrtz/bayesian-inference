"""Module providing different stepsize adaptation kernels."""
import tensorflow_probability as tfp
import functools


class StepSizeAdaptationKernel:
    """Base class for step size adaptation kernels."""

    def __init__(self,
                 num_adaptation_steps,
                 target_accept_prob,
                 step_size_setter_fn,
                 step_size_getter_fn,
                 log_accept_prob_getter_fn):
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
    """Implements the SimpleStepSizeAdaptation kernel.

    Note: This is a wrapper around `tfp.mcmc.SimpleStepSizeAdaptation`.
    """

    def __init__(self,
                 num_adaptation_steps,
                 target_accept_prob=0.75,
                 adaptation_rate=0.01,
                 step_size_setter_fn=tfp.mcmc.simple_step_size_adaptation.hmc_like_step_size_setter_fn,
                 step_size_getter_fn=tfp.mcmc.simple_step_size_adaptation.hmc_like_step_size_getter_fn,
                 log_accept_prob_getter_fn=tfp.mcmc.simple_step_size_adaptation.hmc_like_log_accept_prob_getter_fn,
                 validate_args=False,
                 name=None):
        super(SimpleStepSizeAdaptation, self).__init__(num_adaptation_steps, target_accept_prob,
                                                       step_size_setter_fn, step_size_getter_fn,
                                                       log_accept_prob_getter_fn)
        self.adaptation_rate = adaptation_rate
        self.kernel = functools.partial(tfp.mcmc.SimpleStepSizeAdaptation,
                                        num_adaptation_steps=num_adaptation_steps,
                                        target_accept_prob=target_accept_prob,
                                        adaptation_rate=adaptation_rate,
                                        step_size_setter_fn=step_size_setter_fn,
                                        step_size_getter_fn=step_size_getter_fn,
                                        log_accept_prob_getter_fn=log_accept_prob_getter_fn,
                                        validate_args=validate_args,
                                        name=name)

    def __call__(self, inner_kernel):
        return self.kernel(inner_kernel)


class DualAveragingStepSizeAdaptation(StepSizeAdaptationKernel):
    """Implements the DualAveragingStepSizeAdaptation kernel.

    Note: This is a wrapper around `tfp.mcmc.DualAveragingStepSizeAdaptation`.
    """

    def __init__(self,
                 num_adaptation_steps,
                 target_accept_prob=0.75,
                 exploration_shrinkage=0.05,
                 shrinkage_target=None,
                 step_count_smoothing=10,
                 decay_rate=0.75,
                 step_size_setter_fn=tfp.mcmc.simple_step_size_adaptation.hmc_like_step_size_setter_fn,
                 step_size_getter_fn=tfp.mcmc.simple_step_size_adaptation.hmc_like_step_size_getter_fn,
                 log_accept_prob_getter_fn=tfp.mcmc.simple_step_size_adaptation.hmc_like_log_accept_prob_getter_fn,
                 validate_args=False,
                 name=None):
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
                                        decay_rate=decay_rate,
                                        step_size_setter_fn=step_size_setter_fn,
                                        step_size_getter_fn=step_size_getter_fn,
                                        log_accept_prob_getter_fn=log_accept_prob_getter_fn,
                                        validate_args=validate_args,
                                        name=name)

    def __call__(self, inner_kernel):
        return self.kernel(inner_kernel)
