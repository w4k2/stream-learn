import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import median_filter
from math import e


class CDDD:
    def __init__(self, distance_p=2, filter_size=(3)):
        self.distance_p = distance_p
        self.filter_size = filter_size

        self.iterator = 0
        self.centroids = None
        self.cd_idx = {}
        self.mean_distances = {}
        self.distances = {}
        self.con_array = {}
        self.con = {}
        self.warning = {}
        self.concepts = {}
        self.is_drift = {}

        self._is_drift = False

    def feed(self, X):

        # Unsupervised
        y = np.zeros((X.shape[0]))
        c = [0]

        # If there are no centroids
        if self.centroids is None:
            self.centroids = {}

            # Prepare variables for every class
            for idx in c:
                self.centroids[idx] = [np.mean(X[y == idx], axis=0).tolist()]
                self.cd_idx[idx] = 0
                self.mean_distances[idx] = [0]
                self.con_array[idx] = []
                self.con[idx] = {}
                self.warning[idx] = False
                self.concepts[idx] = []
                self.is_drift[idx] = False
            self.iterator += 1
            return False

        # For each class
        for idx in c:

            # Check if class in actual chunk
            if idx not in np.unique(y):
                # Store last centroid for missing class
                self.centroids[idx].append(self.centroids[idx][-1])
            else:
                # Store actual centroid
                self.centroids[idx].append(np.mean(X[y == idx], axis=0).tolist())

            # Calculate actual distances
            distances = cdist(self.centroids[idx][self.cd_idx[idx]:self.iterator], self.centroids[idx][self.cd_idx[idx]:self.iterator], "cityblock")
            distances = median_filter(distances, size=(self.filter_size))

            # Calculate previous distances
            dist = cdist([self.centroids[idx][self.iterator]], self.centroids[idx][self.cd_idx[idx]:self.iterator], "cityblock")[0]
            dist = median_filter(dist, size=(self.filter_size))

            # Store distances
            self.distances[idx] = distances

            # Calculate mean of distances
            mn = np.mean(dist)

            # Calculate alpha
            x = self.iterator-self.cd_idx[idx]-10
            alpha = 2+(1/(1+e**(x*0.5)))

            # Calculate threshold condition
            con = np.mean(self.mean_distances[idx][self.cd_idx[idx]:self.iterator]) + \
                alpha*np.std(self.mean_distances[idx][self.cd_idx[idx]:self.iterator])

            # Store mean and condition value for visualization
            self.mean_distances[idx].append(mn)
            self.con_array[idx].append(con)

            if self.warning[idx] and mn > self.con[idx] and self.iterator > 3:
                # Drift detected in this class
                self.is_drift[idx] = True
            elif mn > con:
                # Warning status in this class
                self.warning[idx] = True
                self.con[idx] = con
            else:
                # Remove warning status in this class
                self.warning[idx] = False

        if any(self.is_drift.values()):
            # Drift detected
            for idx in self.centroids.keys():
                self.warning[idx] = False
                self.concepts[idx].append(self.iterator)
                self.cd_idx[idx] = self.iterator
                self.is_drift[idx] = False
            self.iterator += 1
            # return True
            self._is_drift = True

        else:
            self.iterator += 1
            # return False
            self._is_drift = False
