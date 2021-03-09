"""Variational inference subpackage.

Modules
-------
vi: module providing the variational inference class.

surrogate_posteriors: module providing different surrogate posterior classes.

flow_bijectors: module providing trainable flow bijectors.
"""

__all___ = ['vi', 'flow_bijectors', 'surrogate_posteriors']

from .vi import VI
