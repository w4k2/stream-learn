import sys
import strlearn
from sklearn import naive_bayes
import arff
import numpy as np
sys.path.insert(0, '../..')


def test_bare_controller():
    with open('datasets/toyset.arff', 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]

    nb_clf = naive_bayes.GaussianNB()
    ctrl = strlearn.controllers.Bare()
    print(ctrl)
    learner = strlearn.Learner(X, y,
                               base_classifier=nb_clf, controller=ctrl)
    learner.run()
    print(ctrl)


def test_budget_controller():
    with open('datasets/toyset.arff', 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]
    nb_clf = naive_bayes.GaussianNB()
    ctrl = strlearn.controllers.Budget()
    print(ctrl)
    learner = strlearn.Learner(X, y,
                               base_classifier=nb_clf, controller=ctrl)
    learner.run()
    print(ctrl)


def test_BLALC_controller():
    with open('datasets/toyset.arff', 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]
    nb_clf = naive_bayes.GaussianNB()
    ctrl = strlearn.controllers.BLALC()
    print(ctrl)
    learner = strlearn.Learner(X, y,
                               base_classifier=nb_clf, controller=ctrl)
    print(learner)
    learner.run()
