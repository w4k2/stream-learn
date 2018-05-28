from sklearn import naive_bayes
from sklearn import neural_network
import sys
import os
import strlearn
import warnings
import arff
import numpy as np
import strlearn as sl
warnings.simplefilter('ignore', DeprecationWarning)

sys.path.insert(0, '../..')

def test_active():
    clfs = {
        "MLP100": neural_network.MLPClassifier(hidden_layer_sizes=(100,)),
        "MLP50": neural_network.MLPClassifier(hidden_layer_sizes=(50,)),
        "MLP101010": neural_network.MLPClassifier(hidden_layer_sizes=(10, 10, 10)),
        "MLP7": neural_network.MLPClassifier(hidden_layer_sizes=(7,))
    }
    tresholds = [.1, .3, .5, .7, .9]
    budgets = [.1, .3, .5, .7, .9]

    controllers = []

    controllers.append(sl.controllers.Bare())

    for budget in budgets:
        controllers.append(sl.controllers.Budget(budget=budget))

    """
    for budget in budgets:
        for treshold in tresholds:
            controllers.append(sl.controllers.BLALC(
                budget=budget, treshold=treshold))
    """

    X, y = None, None
    with open('datasets/toyset.arff', 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]
    for controller in controllers:
            learner = sl.Learner(X, y, clfs["MLP100"],
                                 evaluate_interval=1000, chunk_size=500,
                                 controller=controller)
            learner.run()
            print(learner.scores)
            str(learner)
