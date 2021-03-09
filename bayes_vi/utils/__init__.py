"""Package providing various utility functions and classes.

Modules
-------
datasets: module providing utility function for constructing datasets.

symplectic_integrators: module providing symplectic integrators.
"""

__all__ = ['datasets', 'symplectic_integrators', 'compose', 'to_ordered_dict']

import collections
import functools


def compose(fns):
    """Creates a function composition."""

    def composition(*args, fns_):
        res = fns_[0](*args)
        for f in fns_[1:]:
            res = f(*res)
        return res

    return functools.partial(composition, fns_=fns)


def to_ordered_dict(param_names, state):
    """Transforms list of state parts into collections.OrderedDict with given param names."""
    return collections.OrderedDict([(k, v) for k, v in zip(param_names, state)])
