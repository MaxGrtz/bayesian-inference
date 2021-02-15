import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


class SampleResult:
    """Wrapper for mcmc-like sample results.

    This class precomputes various statistics and mcmc diagnostics.

    Attributes
    ----------
    trace: `list` of `tf.Tensor`
        Traced values defined in the Transition kernels.
    accept_ratios: `list` of `tf.Tensor`
        Per chain ratio of accepted samples in the sampling process.
    """

    def __init__(self, samples, trace):
        """Initializes SampleResult.

        Parameters
        ----------
        samples: `list` of `tf.Tensor`
            The sample results from MCMC or MCMC-like sampling algorithm.
        trace: `list` of `tf.Tensor`
            Traced values defined in the Transition kernels.
        """
        self.samples = samples
        self.trace = trace

        # summary statistics
        if self.trace is not None:
            self.accept_ratios = tf.reduce_mean(tf.cast(self.trace[0], tf.float32), axis=0)

