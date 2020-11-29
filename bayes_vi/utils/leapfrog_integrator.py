import tensorflow as tf


class LeapfrogIntegrator:

    def __init__(self):
        pass

    @tf.function
    def solve(self, hamiltonian_fn, initial_state, step_sizes, num_leapfrog_steps):

        # unpack initial state into initial position q and initial momentum p
        try:
            q, p = tf.split(initial_state, num_or_size_splits=2, axis=-1)
        except tf.errors.InvalidArgumentError:
            raise ValueError('Not a valid state, has to have even event dimension!; was {}'.format(initial_state.shape))

        # initial half step update for momenta
        p = p - 0.5 * step_sizes * tf.gradients(hamiltonian_fn(tf.concat([q, p], axis=-1)), q)[0]

        # main integration loop
        for n in range(num_leapfrog_steps):

            q = q + step_sizes * tf.gradients(hamiltonian_fn(tf.concat([q, p], axis=-1)), p)[0]

            if n != num_leapfrog_steps - 1:
                p = p - step_sizes * tf.gradients(hamiltonian_fn(tf.concat([q, p], axis=-1)), q)[0]

        # final half step for momenta
        p = p - 0.5 * step_sizes * tf.gradients(hamiltonian_fn(tf.concat([q, p], axis=-1)), q)[0]

        # pack final position q and final momentum p to final state
        return tf.concat([q, p], axis=-1)
