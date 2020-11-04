from typing import List
import tensorflow as tf
import tensorflow_probability as tfp

from bayes_vi.utils.functions import apply_fn, apply_fns, apply_inverse_bijector, apply_inverse_bijectors

tfd = tfp.distributions
tfb = tfp.bijectors


class TransformReshapeSplit(tfb.Bijector):

    def __init__(self,
                 transforming_bijectors: List[tfb.Bijector],
                 reshaping_bijectors: List[tfb.Reshape],
                 split_bijector: tfb.Split,
                 validate_args: bool = False,
                 name: str = 'transform_reshape_split'):
        super(TransformReshapeSplit, self).__init__(forward_min_event_ndims=1,
                                                    inverse_min_event_ndims=1,
                                                    validate_args=validate_args,
                                                    name=name)
        self.transforming_bijectors = transforming_bijectors
        self.transform_forward = tfp.mcmc.transformed_kernel.make_transform_fn(self.transforming_bijectors, 'forward')
        self.transform_inverse = tfp.mcmc.transformed_kernel.make_transform_fn(self.transforming_bijectors, 'inverse')
        self.reshaping_bijectors = reshaping_bijectors
        self.reshaping_forward = tfp.mcmc.transformed_kernel.make_transform_fn(self.reshaping_bijectors, 'forward')
        self.reshaping_inverse = tfp.mcmc.transformed_kernel.make_transform_fn(self.reshaping_bijectors, 'inverse')
        self.split_bijector = split_bijector

    def forward(self, x):
        xs = self.split_bijector.forward(x)
        return [*self.transform_forward(self.reshaping_forward(xs))]

    def inverse(self, ys):
        xs = self.reshaping_inverse(self.transform_inverse(ys))
        return self.split_bijector.inverse(xs)

    def inverse_log_det_jacobian(self, y):
        return 0.

    def forward_log_det_jacobian(self, x):
        return 0.


    @classmethod
    def _is_increasing(cls, **kwargs):
        pass
