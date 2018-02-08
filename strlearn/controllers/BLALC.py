from sklearn.exceptions import NotFittedError
from sklearn import preprocessing
import numpy as np

class BLALC(object):
    def __init__(self, budget = .5, treshold = .5):
        self.learner = None
        self.budget = budget
        self.treshold = treshold

    def prepare(self):
        self.pool = int(self.learner.chunk_size * self.budget)
        self.used_pool = 0
        self.used_pools = []
        self.m = self.learner.number_of_classes - 1

    def get_measures(self):
        accumulator = self.used_pools
        self.used_pools = []
        return np.sum(accumulator)

    def next_chunk(self):
        #print "UWAGA CHUNK"
        #print self.learner
        #print self.learner.chunk
        #np.random.shuffle(self.learner.chunk)
        #TODO: Popraw przemieszanie
        self.used_pools.append(self.used_pool)
        self.used_pool = 0

    def should_include(self, X, x, label):
        # Pass first chunk unchanged
        decision = True

        # Analyse
        if self.used_pool > self.pool:
            decision = False
        else:
            try:
                support_vector = self.learner.clf.predict_proba([x])
                normalized_support = preprocessing.normalize(support_vector)
                max_support = np.max(normalized_support)
                value = np.sum(
                    [abs(support-max_support) for support in normalized_support]
                    ) / (self.learner.number_of_classes - 1)
                if value < self.treshold:
                    decision = True
                    self.used_pool += 1
                else:
                    decision = False
            except NotFittedError as e:
                decision = True

        return decision

    def should_break_chunk(self, X):
        return self.used_pool > self.pool

    def __str__(self):
        return 'blalc_b%.2f_t%.2f' % (self.budget, self.treshold)
