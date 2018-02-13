import sys, os, strlearn
sys.path.insert(0, '../..')

from sklearn import naive_bayes

def test_bare_controller():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    ctrl = strlearn.controllers.Bare()
    learner = strlearn.Learner(stream = toystream, base_classifier = nb_clf, controller = ctrl)
    learner.run()
    print ctrl

def test_budget_controller():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    ctrl = strlearn.controllers.Budget()
    learner = strlearn.Learner(stream = toystream, base_classifier = nb_clf, controller = ctrl)
    learner.run()
    print ctrl

def test_BLALC_controller():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    ctrl = strlearn.controllers.BLALC()
    learner = strlearn.Learner(stream = toystream, base_classifier = nb_clf, controller = ctrl)
    learner.run()
    print ctrl
