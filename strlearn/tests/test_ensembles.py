from sklearn import naive_bayes
import sys
import os
import strlearn
import warnings
import strlearn

sys.path.insert(0, '../..')
warnings.simplefilter(action='ignore', category=FutureWarning)


def test_WAE():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        clf = strlearn.ensembles.WAE()
        clf.set_base_clf(nb_clf)
        learner = strlearn.Learner(toystream, clf)
        learner.run()
        str(learner)


def test_pp_WAE():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        clf = strlearn.ensembles.WAE(post_pruning=True)
        clf.set_base_clf(nb_clf)
        learner = strlearn.Learner(toystream, clf)
        learner.run()
        learner.serialize('ppWAE.csv')
        os.remove('ppWAE.csv')


def test_WAE_wcm():
    methods = ('same_for_each', 'proportional_to_accuracy', 'kuncheva',
               'proportional_to_accuracy_related_to_whole_ensemble',
               'bell_curve')
    for method in methods:
        nb_clf = naive_bayes.GaussianNB()
        with open('datasets/toyset.arff', 'r') as toystream:
            clf = strlearn.ensembles.WAE(weight_calculation_method=method)
            clf.set_base_clf(nb_clf)
            learner = strlearn.Learner(toystream, clf)
            learner.run()


def test_WAE_am():
    methods = ('weights_proportional', 'constant', 'gaussian')
    for method in methods:
        nb_clf = naive_bayes.GaussianNB()
        with open('datasets/toyset.arff', 'r') as toystream:
            clf = strlearn.ensembles.WAE(aging_method=method)
            clf.set_base_clf(nb_clf)
            learner = strlearn.Learner(toystream, clf)
            learner.run()


def test_pp_WAE_rejuvenation():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        clf = strlearn.ensembles.WAE(rejuvenation_power=.5,
                                     post_pruning=True)
        clf.set_base_clf(nb_clf)
        learner = strlearn.Learner(toystream, clf)
        learner.run()
        learner.serialize('ppWAE.csv')
        os.remove('ppWAE.csv')


def test_WAE_rejuvenation():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        clf = strlearn.ensembles.WAE(rejuvenation_power=.5)
        clf.set_base_clf(nb_clf)
        learner = strlearn.Learner(toystream, clf)
        learner.run()
        learner.serialize('ppWAE.csv')
        os.remove('ppWAE.csv')


def test_REA():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        clf = strlearn.ensembles.REA(base_classifier=nb_clf)
        learner = strlearn.Learner(toystream, clf)
        learner.run()
        learner.serialize('REA.csv')
        os.remove('REA.csv')
