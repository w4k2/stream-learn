import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from scipy.stats import hmean

_SQRT2 = np.sqrt(2)

class SDDE(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=3, immobilizer=5, n_detectors=15, subspace_size=1, random_state=None, sensitivity=0.55):
        self.immobilizer = immobilizer
        self.sigma = sigma
        self.n_detectors = n_detectors
        self.subspace_size = subspace_size
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)

        self.count = 0
        self.confidence=[]
        self.sensitivity = sensitivity

        self.drf_threshold = int((self.n_detectors*2)*self.sensitivity)
            
    def feed(self, X, y, pred):

        self.count+=1

        # Init: tabele, wielkość podprzestrzeni, losowanie podprzestrzenie
        if not hasattr(self, "drift"):
            self.drift = []
            self.all_elements = []
            self.base_kernels = []

            n_features = X.shape[1]

            [self.all_elements.append([]) for i in range(2*self.n_detectors)]
            [self.base_kernels.append([]) for i in range(self.n_detectors)]

            if self.subspace_size == 'auto':
                self.subspace_size = np.ceil(np.sqrt(n_features)).astype(int)

            self.subspaces = np.array([self.random.choice(list(range(n_features)),
                                                        size=self.subspace_size,
                                                        replace=False)
                                       for _ in range(self.n_detectors)])

        # Zbiór kerneli dla chunka
        self.temp_kernels = []

        # Zbiór na tdm, cmcd dla chunka
        chunk_elements = []

        # Dla kadej z podprzestrzeni 
        for sub_id, subspace in enumerate(self.subspaces):
            
            samples = [
                X[:,subspace],
                X[:,subspace][y==0],
                X[:,subspace][y==1],
                [y]
            ]
            sources = [
                X[:,subspace],
                X[:,subspace],
                X[:,subspace],
                [y]
            ]

            kernels = [KernelDensity().fit(sample) for sample in samples]
            self.temp_kernels.append(kernels)

            # Bazowe kernele dla każdego podzbioru cech tylko raz
            if len(self.base_kernels[sub_id]) == 0:
                self.base_kernels[sub_id] = kernels

            self.cf_s = [[
                np.exp(kernel.score_samples(source))
                for kernel, source in zip(k, sources)]
                    for k in (kernels, self.base_kernels[sub_id])]

            # Dzieła zebrane
            el1 = self._tdm(*self.cf_s)
            el2 = self._cmcd(*self.cf_s)
            chunk_elements.append([el1,el2])

        # Zebranie wszystkiego do ciaglej listy
        chunk_elements = np.squeeze(np.array(chunk_elements).reshape(1,-1)).tolist()

        # Integracja
        if len(self.drift) > self.immobilizer:

            # shape: detectors x (tdm, cmcd) x chunks
            self.combined_elements = np.array(self.all_elements).reshape(self.n_detectors, 2, -1) 

            is_drift_arr = np.array([self._is_drift(el, hmean(self.combined_elements[:,el_idx%2,:], axis=0))
                                        for el_idx, el in enumerate(chunk_elements)])

            drf_cnt = np.sum(is_drift_arr)
            self.confidence.append(drf_cnt)

            # Detekcja
            if drf_cnt >= self.drf_threshold:
                self.drift.append(2)
                self.base_kernels = self.temp_kernels
            else:
                self.drift.append(0)
        else:
            self.drift.append(0)
            self.confidence.append(0)
        
        # Dodanie wyników z chunka do wszystkich
        for id, el in enumerate(chunk_elements):
            self.all_elements[id].append(el)

        return self

    def _is_drift(self, el, els):
        
        s = np.std(els)

        if s==0:
            els = np.copy(els[1:])
            bas = np.power(10,-(np.mean(np.log(els))//2))
            s = np.std(els*bas)/bas
        
        return np.abs(el - np.mean(els)) > s * self.sigma

    def _hellinger(self, p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    def _tdm(self, f_s, c_s):
        return self._hellinger(f_s[0], c_s[0])

    def _cmcd(self, f_s, c_s):
        return np.sum([
            ((f_s[3] + c_s[3])/2) * .5 * np.sum(np.abs(f_s[1+i] - c_s[1+i]), axis=0) for i in range(2)
        ])