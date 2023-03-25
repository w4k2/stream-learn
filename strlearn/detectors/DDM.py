import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class DDM(BaseEstimator, ClassifierMixin):
    def __init__(self, warning_lvl=2, drift_lvl=3, skip=30):
        self.warning_lvl = warning_lvl
        self.drift_lvl = drift_lvl
        self.skip = skip

        self.p_min = np.inf
        self.s_min = np.inf
        self.p_i = 1
        self.s_i = 0
        self.cnt = 1

        self.drift = []

    def feed(self, X, real, pred):

        if len(pred) != len(real):
            self.drift.append(0)
            return self

        chunk_p_i = []
        chunk_s_i = []
        chunk_p_mins = []
        chunk_s_mins = []

        err = np.abs(real-pred)
        for e in err:
            self.p_i += (e - self.p_i)/self.cnt
            self.s_i = np.sqrt(self.p_i * (1- self.p_i)/self.cnt)
            self.cnt+=1

            chunk_p_i.append(self.p_i)
            chunk_s_i.append(self.s_i)

            if self.p_i + self.s_i < self.p_min + self.s_min:
                self.p_min = self.p_i
                self.s_min = self.s_i

            chunk_p_mins.append(self.p_min)
            chunk_s_mins.append(self.s_min)

        if self.cnt > self.skip and self.p_i + self.s_i > self.p_min + self.drift_lvl*self.s_min:
            self.drift.append(2)
            # reset
            self.p_min = np.inf
            self.s_min = np.inf
            self.p_i = 1
            self.s_i = 0
            self.cnt = 1
        elif self.cnt > self.skip and self.p_i + self.s_i >= self.p_min + self.warning_lvl*self.s_min:
            self.drift.append(1)
        else:
            self.drift.append(0)

        return self