class EpsilonScheduler:
    def __init__(self, eps_start=1.0, eps_min=0.05, decay=0.985):
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.decay = decay

    def step(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.decay)
        return self.epsilon
