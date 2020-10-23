from bayes_vi.mcmc import MCMC
from bayes_vi.vi import VI


class InferenceObject:

    def __init__(self, model, y, x, constraining_bijectors=None):
        self.model = model
        self.y = y
        self.x = x
        self.constraining_bijectors = constraining_bijectors

    def mcmc(self, transition_kernel):
        return MCMC(self.model, self.y, self.x, self.constraining_bijectors, transition_kernel)

    def vi(self):
        return VI(self.model, self.y, self.x, self.constraining_bijectors)
