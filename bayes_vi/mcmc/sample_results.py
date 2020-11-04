from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


class SampleResult:

    def __init__(self, model, samples, trace):
        self.model = model
        self.samples = samples
        self.trace = trace

        # summary statistics
        self.accept_ratios = tf.reduce_mean(tf.cast(self.trace[0], tf.float32), axis=0)

        self.r_hat = tfp.mcmc.potential_scale_reduction(self.samples)

        self.eff_ss = [tfp.mcmc.effective_sample_size(part, cross_chain_dims=1 if part.shape[1] > 1 else None)
                       for part in self.samples]

        self.auto_correlations = [tfp.stats.auto_correlation(part, axis=0)
                                  for part in self.samples]

        self.mean, self.variance = map(list, zip(*[tf.nn.moments(part, axes=[0, 1])
                                                   for part in self.samples]))

        self.stddev = [tf.sqrt(var) for var in self.variance]

        self.mcse = [tf.sqrt(var / ess) for var, ess in zip(self.variance, self.eff_ss)]

        self.percentiles = [tfp.stats.percentile(part,
                                                 q=[0, 2.5, 50, 97.5, 100],
                                                 interpolation='linear',
                                                 axis=[0, 1])
                            for part in self.samples]

    def sample_statistics(self, format_as: str = 'namedtuple'):
        sample_stats = [[np.round(mean.numpy(), 3),
                         np.round(stddev.numpy(), 3),
                         np.round(mcse.numpy(), 3),
                         *(np.round(perc.numpy(), 3) for perc in tf.unstack(percentiles)),
                         np.round(r_hat.numpy(), 3),
                         np.round(eff_ss.numpy(), 0)]
                        for mean, stddev, mcse, percentiles, r_hat, eff_ss
                        in zip(self.mean,
                               self.stddev,
                               self.mcse,
                               self.percentiles,
                               self.r_hat,
                               self.eff_ss)]

        SampleStatistics = namedtuple('SampleStatistics', ['param', 'mean', 'stddev', 'mcse',
                                                           'min', 'HDI_95_lower', 'mode', 'HDI_95_upper', 'max',
                                                           'R_hat', 'eff_ss'])
        sample_statistics_tuples = [SampleStatistics._make([param, *stats])
                                    for param, stats in zip(self.model.param_names, sample_stats)]
        if format_as == 'namedtuple':
            return sample_statistics_tuples
        elif format_as == 'df':
            return pd.DataFrame.from_records(
                sample_statistics_tuples,
                columns=SampleStatistics._fields,
                index='param'
            )
