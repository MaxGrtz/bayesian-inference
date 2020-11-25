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
    model: `Model`
        A Bayesian probabilistic `Model`.
    samples: `list` of `tf.Tensor`
        The sample results from MCMC or MCMC-like sampling algorithm.
    trace: `list` of `tf.Tensor`
        Traced values defined in the Transition kernels.
    accept_ratios: `list` of `tf.Tensor`
        Per chain ratio of accepted samples in the sampling process.
    r_hat: `tf.Tensor`
        MCMC convergence diagnostic `tfp.mcmc.potential_scale_reduction` for each parameter.
    eff_ss: `tf.Tensor`
        effective samples for each parameter based on auto-correlations `tfp.mcmc.effective_sample_size`.
    auto_correlations: `list` of `tf.Tensor`
        per chain auto-correlations of the samples of each parameter.
    mean: `tf.Tensor`
        mean of each parameter across all chains.
    variance: `tf.Tensor`
        variance of each parameter across all chains.
    stddev: `tf.Tensor`
        stddev of each parameter across all chains.
    mcse: `tf.Tensor`
        Monte-Carlo standard error for each parameter as tf.sqrt(variance / eff_ss)
    percentiles: `list` of `tf.Tensor`
        percentiles [0, 2.5, 50, 97.5, 100] for each parameter across all chains.
    """

    def __init__(self, model, samples, trace):
        """Initializes SampleResult.

        Parameters
        ----------
        model: `Model`
            A Bayesian probabilistic `Model`.
        samples: `list` of `tf.Tensor`
            The sample results from MCMC or MCMC-like sampling algorithm.
        trace: `list` of `tf.Tensor`
            Traced values defined in the Transition kernels.
        """
        self.model = model
        self.samples = samples
        self.trace = trace

        # summary statistics
        if self.trace is not None:
            self.accept_ratios = tf.reduce_mean(tf.cast(self.trace[0], tf.float32), axis=0)

        # compute r_hat, set to 0. if num_chains == 1
        if self.samples[0].shape[1] > 1:
            self.r_hat = tfp.mcmc.potential_scale_reduction(self.samples)
        else:
            self.r_hat = [tf.zeros(shape=part.shape[2:]) for part in self.samples]

        # compute effective sample size for each state part, cross chain if num_chains > 1
        self.eff_ss = [
            tf.squeeze(
                tfp.mcmc.effective_sample_size(
                    part, cross_chain_dims=1 if part.shape[1] > 1 else None
                )
            )
            for part in self.samples]

        # compute auto-correlations for each state part and chain
        self.auto_correlations = [tfp.stats.auto_correlation(part, axis=0)
                                  for part in self.samples]

        # compute means and variances for each state part across all chains
        self.mean, self.variance = map(list, zip(*[tf.nn.moments(part, axes=[0, 1])
                                                   for part in self.samples]))

        # derive stddev from variance
        self.stddev = [tf.sqrt(var) for var in self.variance]

        # compute monte-carlo standard error for each state part
        self.mcse = [tf.sqrt(var / ess) for var, ess in zip(self.variance, self.eff_ss)]

        # compute percentiles across all chains
        self.percentiles = [tfp.stats.percentile(part,
                                                 q=[0, 2.5, 50, 97.5, 100],
                                                 interpolation='linear',
                                                 axis=[0, 1])
                            for part in self.samples]

    def sample_statistics(self, format_as: str = 'namedtuple'):
        """Computes and formats sample statistics."""
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

        SampleStatistics = collections.namedtuple(
            'SampleStatistics',
            ['param', 'mean', 'stddev', 'mcse',
             'min', 'HDI_95_lower', 'mode', 'HDI_95_upper', 'max',
             'R_hat', 'eff_ss'])
        sample_statistics_tuples = [SampleStatistics._make([name, *flat_stat])
                                    for param, stats in zip(self.model.param_names, sample_stats)
                                    for name, flat_stat in zip(*self.flatten_stats(param, stats))]
        if format_as == 'namedtuple':
            return sample_statistics_tuples
        elif format_as == 'df':
            return pd.DataFrame.from_records(
                sample_statistics_tuples,
                columns=SampleStatistics._fields,
                index='param'
            )

    @staticmethod
    def flatten_stats(param, stats):
        """The `stats` of `param` are flattened and marked by indices."""
        arr = stats[0]
        if len(arr.shape) != 0:
            indices = np.indices(arr.shape).reshape(len(arr.shape), -1).T
            names = ['{}{}'.format(param, list(idx)) for idx in indices]
            flat_stats = [[arr[tuple(idx)] for arr in stats] for idx in indices]
            return names, flat_stats
        else:
            return [param], [stats]

    def structured_results(self, results):
        """The `results` are wrapped in an collections.OrderedDict`"""
        events = [r[0, 0] for r in results]
        results = [r.numpy() for r in results]
        structured_results = collections.OrderedDict([])
        for param, event, result in zip(self.model.param_names, events, results):
            if len(event.shape) != 0:
                indices = np.indices(event.shape).reshape(len(event.shape), -1).T
                names = ['{}{}'.format(param, list(idx)) for idx in indices]
                flat_result = [result[(slice(None), slice(None)) + tuple(idx)] for idx in indices]
                structured_result = collections.OrderedDict([
                    (name, collections.OrderedDict([
                        ('chain-{}'.format(c), v) for c, v in enumerate(s.T)
                    ]))
                    for name, s in zip(names, flat_result)
                ])
                structured_results.update([(param, structured_result)])
            else:
                structured_result = collections.OrderedDict([
                    (param, collections.OrderedDict([
                        ('chain-{}'.format(c), v) for c, v in enumerate(result.T)
                    ]))])
                structured_results.update([(param, structured_result)])
        return structured_results
