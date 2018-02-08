import numpy as np

class Budget(object):
    def __init__(self, budget = .5):
        self.learner = None
        self.budget = budget

    def prepare(self):
        self.pool = int(self.learner.chunk_size * self.budget)
        self.used_pool = 0
        self.used_pools = []

    def get_measures(self):
        accumulator = self.used_pools
        self.used_pools = []
        return np.sum(accumulator)

    def next_chunk(self):
        self.used_pools.append(self.used_pool)
        self.used_pool = 0

    def should_include(self, _X, _x, _label):
        # Analyse
        if self.used_pool > self.pool:
            decision = False
        else:
            decision = True
            self.used_pool += 1

        return decision

    def should_break_chunk(self, _X):
        return False

    def __str__(self):
        return 'bc_b%.2f' % (self.budget)
