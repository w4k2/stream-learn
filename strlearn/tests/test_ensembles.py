import sys
import os
import strlearn
sys.path.insert(0, '../..')

from sklearn import naive_bayes


def test_WAE():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(
        base_classifier=nb_clf,
        pruning_criterion='accuracy',
        weight_calculation_method='proportional_to_accuracy_related_to_whole_ensemble',
        rejuvenation_power = .5,
        is_post_pruning=True)
    learner = strlearn.Learner(toystream, clf)
    learner.run()


def test_pp_WAE():
    nb_clf = naive_bayes.GaussianNB()
    toystream = open('datasets/toyset.arff', 'r')
    clf = strlearn.ensembles.WAE(base_classifier=nb_clf, is_post_pruning=True)
    learner = strlearn.Learner(toystream, clf)
    learner.run()
    learner.serialize('ppWAE.csv')
    os.remove('ppWAE.csv')
