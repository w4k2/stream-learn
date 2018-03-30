from sklearn import naive_bayes
import sys
import os
import strlearn
import warnings

import strlearn

sys.path.insert(0, '../..')
warnings.simplefilter('always')


def test_WAE():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(base_classifier=nb_clf)
    learner = strlearn.Learner(toystream, clf)
    learner.run()
    str(learner)

"""

def test_pp_WAE():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(base_classifier=nb_clf, is_post_pruning=True)
    learner = strlearn.Learner(toystream, clf)
    learner.run()
    learner.serialize('ppWAE.csv')
    os.remove('ppWAE.csv')


def test_WAE_wcm():
    methods = ('same_for_each', 'kuncheva',
               'proportional_to_accuracy_related_to_whole_ensemble',
               'proportional_to_accuracy_related_to_whole_ensemble_using_bell_curve')
    for method in methods:
        nb_clf = naive_bayes.GaussianNB()
        toystream = open('datasets/toyset.arff', 'r')
        clf = strlearn.ensembles.WAE(base_classifier=nb_clf,
                                     weight_calculation_method=method)
        learner = strlearn.Learner(toystream, clf)
        learner.run()


def test_WAE_am():
    methods = ('weights_proportional', 'constant', 'gaussian')
    for method in methods:
        nb_clf = naive_bayes.GaussianNB()
        toystream = open('datasets/toyset.arff', 'r')
        clf = strlearn.ensembles.WAE(base_classifier=nb_clf, aging_method=method)
        learner = strlearn.Learner(toystream, clf)
        learner.run()


def test_pp_WAE_rejuvenation():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(base_classifier=nb_clf, rejuvenation_power=.5, is_post_pruning=True)
    learner = strlearn.Learner(toystream, clf)
    learner.run()
    learner.serialize('ppWAE.csv')
    os.remove('ppWAE.csv')


def test_WAE_rejuvenation():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(base_classifier=nb_clf, rejuvenation_power=.5)
    learner = strlearn.Learner(toystream, clf)
    learner.run()
    learner.serialize('ppWAE.csv')
    os.remove('ppWAE.csv')
"""

def test_REA():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.REA(base_classifier=nb_clf)
    learner = strlearn.Learner(toystream, clf)
    learner.run()
    learner.serialize('REA.csv')
    os.remove('REA.csv')
