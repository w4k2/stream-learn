from sklearn import naive_bayes
from sklearn import neural_network
import sys
import os
import strlearn
import warnings
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

    for budget in budgets:
        for treshold in tresholds:
            controllers.append(sl.controllers.BLALC(
                budget=budget, treshold=treshold))

    for controller in controllers:
        with open('datasets/toyset.arff', 'r') as toystream:
            learner = sl.Learner(toystream, clfs["MLP100"],
                                 evaluate_interval=1000, chunk_size=500,
                                 controller=controller)
            learner.run()
            print(learner.scores)
            str(learner)
    assert(False)
