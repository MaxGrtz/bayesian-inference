import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class SurrogatePosterior:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Not yet implemented!')


class MeanFieldADVI(SurrogatePosterior):

    def __init__(self):
        super(MeanFieldADVI, self).__init__()

    def __call__(self, model):
        # Sample to get a list of Tensors
        list_of_samples = list(model.prior_distribution.sample().values())

        distlist = []
        for i, (sample, bijector, name) in enumerate(zip(list_of_samples, model.constraining_bijectors, model.param_names)):
            dtype = sample.dtype
            rv_shape = sample[0].shape
            loc = tf.Variable(
                tf.random.normal(rv_shape, dtype=dtype),
                name='meanfield_%s_mu' % i,
                dtype=dtype)
            scale = tfp.util.TransformedVariable(
                tf.fill(rv_shape, value=tf.constant(0.02, dtype)),
                tfb.Softplus(),
                name='meanfield_%s_scale' % i,
            )
            approx_node = tfd.TransformedDistribution(tfd.Normal(loc=loc, scale=scale), bijector=bijector)

            distlist.append((name, approx_node))


        return tfd.JointDistributionNamedAutoBatched(distlist)


class NormalizingFlow(SurrogatePosterior):

    def __init__(self, base_distributions, flow_bijectors):
        super().__init__()
        self.base_distributions = base_distributions
        self.flow_bijectors = flow_bijectors

    def __call__(self, model, constraining_bijectors):
        # Sample to get a list of Tensors
        list_of_samples = list(model.joint_distribution.sample(1).values())[:-1]

        distlist = []
        for i, (sample, base_dist, flow_bijector, constr_bijector) \
                in enumerate(
            zip(list_of_samples, self.base_distributions, self.flow_bijectors, constraining_bijectors)):

            rv_shape = sample[0].shape

            transformed = tfd.TransformedDistribution(
                distribution=base_dist,
                bijector=flow_bijector,
                event_shape=(1,)
            )

            approx_node = tfd.TransformedDistribution(transformed, bijector=constr_bijector)

            if rv_shape == ():
                distlist.append(approx_node)
            else:
                distlist.append(
                    tfd.Independent(approx_node, reinterpreted_batch_ndims=1)
                )

        return tfd.JointDistributionSequential(distlist)
