import tensorflow as tf
import tensorflow_probability as tfp
import functools

tfb = tfp.bijectors


def make_val_and_grad_fn(value_fn):
    @functools.wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)
    return val_and_grad


def apply_inverse_bijectors(bijectors, ys):
    return list(map(lambda bij, y: bij.inverse(y), bijectors, ys))


def apply_inverse_bijector(bijector, ys):
    return list(map(lambda bij, y: bij.inverse(y), [bijector]*len(ys), ys))


def apply_fns(fns, xs):
    return list(map(lambda f, y: f(y), fns, xs))


def apply_fn(fn, xs):
    return list(map(lambda f, y: f(y), [fn] * len(xs), xs))


def compose(fns):
    def composition(*args, fns_):
        res = fns_[0](*args)
        for f in fns_[1:]:
            res = f(*res)
        return res

    return functools.partial(composition, fns_=fns)
