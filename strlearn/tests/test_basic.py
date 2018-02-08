import sys
sys.path.insert(0, '../..')

import strlearn

from sklearn import tree, neural_network


# Testing controllers
def test_bare_controller():
    mlp_clf = neural_network.MLPClassifier()
    toystream = open('datasets/toyset.arff', 'r')
    ctrl = strlearn.controllers.Bare()
    learner = strlearn.Learner(stream = toystream, base_classifier = mlp_clf, controller = ctrl)
    learner.run()
    print ctrl

def test_budget_controller():
    mlp_clf = neural_network.MLPClassifier()
    toystream = open('datasets/toyset.arff', 'r')
    ctrl = strlearn.controllers.Budget()
    learner = strlearn.Learner(stream = toystream, base_classifier = mlp_clf, controller = ctrl)
    learner.run()
    print ctrl

def test_BLALC_controller():
    mlp_clf = neural_network.MLPClassifier()
    toystream = open('datasets/toyset.arff', 'r')
    ctrl = strlearn.controllers.BLALC()
    learner = strlearn.Learner(stream = toystream, base_classifier = mlp_clf, controller = ctrl)
    learner.run()
    print ctrl

def test_WAE():
    dt_clf = tree.DecisionTreeClassifier()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(base_classifier = dt_clf)
    learner = strlearn.Learner(toystream, clf, verbose=False)
    learner.run()


def test_pp_WAE():
    dt_clf = tree.DecisionTreeClassifier()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(base_classifier = dt_clf, is_post_pruning=True)
    learner = strlearn.Learner(toystream, clf, verbose=False)
    learner.run()
    learner.serialize('ppWAE.csv')
