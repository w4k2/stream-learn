import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EDDM(BaseEstimator, ClassifierMixin):
    def __init__(self, warning_lvl = 0.95, drift_lvl = 0.9):
        self.warning_lvl = warning_lvl
        self.drift_lvl = drift_lvl
        self.drift = []

        self.sample_cnt = 0
        self.distances_since_drift = []
        self.last_err_idx = 0

        self.mean_dist = 0
        self.std_dist = 0
        self.mean_dist_max = 0
        self.std_dist_max = 0

    def feed(self, X, real, pred):

        if len(pred) != len(real):
            self.drift.append(0)
            return self
        
        chunk_m_i=[]
        chunk_s_i=[]
        chunk_m_max=[]
        chunk_s_max=[]

        err = np.abs(real-pred)
        for e in err:
            self.sample_cnt+=1
            if(e==1):
                self.last_err_idx = self.sample_cnt

            self.distances_since_drift.append(np.abs(self.last_err_idx-self.sample_cnt))

            m_i = np.mean(np.array(self.distances_since_drift))
            s_i = np.std(np.array(self.distances_since_drift))

            if m_i + 2*s_i > (self.mean_dist_max + 2* self.std_dist_max):
                self.mean_dist_max = m_i
                self.std_dist_max = s_i
            
            chunk_m_i.append(m_i)
            chunk_s_i.append(s_i)
            chunk_m_max.append(self.mean_dist_max)
            chunk_s_max.append(self.std_dist_max)

        if (m_i + 2*s_i)/(self.mean_dist_max + 2* self.std_dist_max)<self.drift_lvl:
            self.drift.append(2)
            #reset            
            self.mean_dist_max = 0
            self.std_dist_max = 0
            self.distances_since_drift=[]
        elif (m_i + 2*s_i)/(self.mean_dist_max + 2* self.std_dist_max)<self.warning_lvl:
            self.drift.append(1)
        else:
            self.drift.append(0)
    
        return self