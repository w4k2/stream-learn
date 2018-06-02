import arff
import numpy as np

def load_arff(path):
    with open(path, 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]
    return X, y
