import functools
import tensorflow_probability as tfp
import tensorflow as tf
from bayes_vi.mcmc.sample_results import SampleResult

tfd = tfp.distributions


class MCMC:

    def __init__(self, model_fn, y, x, constraining_bijectors, transition_kernel):
        self.y = y
        self.x = x
        self.model = tfd.JointDistributionCoroutine(model_fn(y, x))
        self.constraining_bijectors = constraining_bijectors
        self.target_log_prob_fn = lambda *args: self.model.log_prob(args + (y,))
        self.transition_kernel = transition_kernel


    def sample(self, num_chains, num_samples, num_burnin_steps, initial_state=None):

        trace_fns = [self.transition_kernel.trace_fn]

        transformed_kernel = self.transition_kernel(self.target_log_prob_fn)

        if self.constraining_bijectors:
            transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=transformed_kernel, bijector=self.constraining_bijectors)
            trace_fns.append(lambda _, pkr: (_, pkr.inner_results))

        if hasattr(self.transition_kernel, 'step_size_adaptation_kernel'):
            if self.transition_kernel.step_size_adaptation_kernel:
                transformed_kernel = self.transition_kernel.step_size_adaptation_kernel(inner_kernel=transformed_kernel)
                trace_fns.append(lambda _, pkr: (_, pkr.inner_results))

        if initial_state is None:
            initial_state = [part for part in self.model.sample(num_chains)[:-1]]

        trace_fn = self.compose(list(reversed(trace_fns)))

        results = self.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=transformed_kernel,
            trace_fn=trace_fn
        )
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


    @staticmethod
    def compose(fns):
        def composition(*args, fns_):
            res = fns_[0](*args)
            for f in fns_[1:]:
                res = f(*res)
            return res
        return functools.partial(composition, fns_=fns)

