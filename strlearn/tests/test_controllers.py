import sys
import strlearn
from sklearn import naive_bayes
sys.path.insert(0, '../..')


def test_bare_controller():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        ctrl = strlearn.controllers.Bare()
        print(ctrl)
        learner = strlearn.Learner(stream=toystream,
                                   base_classifier=nb_clf, controller=ctrl)
        learner.run()
        print(ctrl)


def test_budget_controller():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        ctrl = strlearn.controllers.Budget()
        print(ctrl)
        learner = strlearn.Learner(stream=toystream,
                                   base_classifier=nb_clf, controller=ctrl)
        learner.run()
        print(ctrl)


def test_BLALC_controller():
    nb_clf = naive_bayes.GaussianNB()
    with open('datasets/toyset.arff', 'r') as toystream:
        ctrl = strlearn.controllers.BLALC()
        print(ctrl)
        learner = strlearn.Learner(stream=toystream,
                                   base_classifier=nb_clf, controller=ctrl)
        print(learner)
        learner.run()
