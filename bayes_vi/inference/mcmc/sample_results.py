"""Module providing wrapper class for MCMC sampling results."""

import tensorflow as tf


class SampleResult:
    """Wrapper for mcmc sample results.

    Attributes
    ----------
    samples: `list` of `tf.Tensor`
        The sample results from MCMC or MCMC-like sampling algorithm.
    trace: `dict[str, tf.Tensor]`
        Mapping from names of traced metrics to traced values defined in the Transition kernels.
    accept_ratios: `list` of `tf.Tensor`
        Per chain ratio of accepted samples in the sampling process.
    """

    def __init__(self, samples, trace):
        """Initializes SampleResult.

        Parameters
        ----------
        samples: `list` of `tf.Tensor`
            The sample results from MCMC or MCMC-like sampling algorithm.
        trace: `dict[str, tf.Tensor]`
            Mapping from names of traced metrics to traced values defined in the Transition kernels.
        """
        self.samples = samples
        self.trace = trace

        # summary statistics
        if self.trace is not None:
            self.accept_ratios = tf.reduce_mean(tf.cast(self.trace['is_accepted'], tf.float32), axis=0)
