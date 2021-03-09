"""MCMC subpackage containing MCMC related modules.

Modules
-------
mcmc: module providing the mcmc inference class

sample_results: module providing a wrapper for the mcmc sampling results.

stepsize_adaptation_kernels: module providing different stepsize adaptation kernels.

transition_kernels: module providing different Markov transition kernels.
"""

__all__ = ['mcmc', 'sample_results', 'stepsize_adaptation_kernels', 'transition_kernels']

from .mcmc import MCMC
