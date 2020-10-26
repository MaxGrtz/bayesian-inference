import tensorflow_probability as tfp
import tensorflow as tf
from collections import namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SampleResult:

    def __init__(self, model, samples, trace):
        self.model = model
        self.samples = [tf.squeeze(part) for part in samples]
        self.trace = trace
        self.accept_ratios = tf.reduce_mean(tf.cast(self.trace[0], tf.float32), axis=0)
        self.potential_scale_reduction = tfp.mcmc.potential_scale_reduction(self.samples)
        self.effective_sample_sizes = tfp.mcmc.effective_sample_size(self.samples)
        self.auto_correlations = tfp.stats.auto_correlation(self.samples, axis=1)
        self.percentiles = tf.transpose(tfp.stats.percentile(self.samples, q=[0, 2.5, 50, 97.5, 100],
                                                             interpolation='linear', axis=[1, 2]))

    def sample_statistics(self, format_as='namedtuple'):
        sample_stats = [[tf.reduce_mean(chains),
                         tfp.stats.stddev(chains, sample_axis=[0, 1]),
                         *tf.unstack(percentiles), r_hat, tf.round(tf.reduce_mean(eff_ss))]
                        for chains, percentiles, r_hat, eff_ss
                        in zip(self.samples,
                               self.percentiles,
                               self.potential_scale_reduction,
                               self.effective_sample_sizes)]

        distributions, _ = self.model.sample_distributions()
        prior_params = [dist.name.split('_')[-1] for dist in distributions[:-1]]
        SampleStatistics = namedtuple('SampleStatistics', ['param', 'mean', 'stddev',
                                                           'min', 'HDI_95_lower', 'mode', 'HDI_95_upper', 'max',
                                                           'R_hat', 'eff_ss'])
        sample_statistics_tuples = [SampleStatistics._make([param] + [np.round(stat.numpy(), 3) for stat in stats])
                                    for param, stats in zip(prior_params, sample_stats)]
        if format_as == 'namedtuple':
            return sample_statistics_tuples
        elif format_as == 'df':
            return pd.DataFrame.from_records(
                sample_statistics_tuples,
                columns=SampleStatistics._fields,
                index='param'
            )

