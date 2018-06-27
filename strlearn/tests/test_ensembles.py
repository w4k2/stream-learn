import sys
import strlearn as sl

sys.path.insert(0, '../..')

def test_WAE():
    X, y = sl.utils.load_arff('toyset.arff')
    clf = sl.ensembles.WAE()
    learner = sl.Learner(X, y, clf)
    learner.run()

def test_pp_WAE():
    X, y = sl.utils.load_arff('toyset.arff')
    clf = sl.ensembles.WAE(post_pruning=True)
    learner = sl.Learner(X, y, clf)
    learner.run()

def test_WAE_wcm():
    X, y = sl.utils.load_arff('toyset.arff')
    methods = ('same_for_each', 'proportional_to_accuracy', 'kuncheva',
               'proportional_to_accuracy_related_to_whole_ensemble',
               'bell_curve')
    for method in methods:
        clf = sl.ensembles.WAE(weight_calculation_method=method)
        learner = sl.Learner(X, y, clf)
        learner.run()


def test_WAE_am():
    X, y = sl.utils.load_arff('toyset.arff')
    methods = ('weights_proportional', 'constant', 'gaussian')
    for method in methods:
        clf = sl.ensembles.WAE(aging_method=method)
        learner = sl.Learner(X, y, clf)
        learner.run()


def test_pp_WAE_rejuvenation():
    X, y = sl.utils.load_arff('toyset.arff')
    clf = sl.ensembles.WAE(rejuvenation_power=.5, post_pruning=True)
    learner = sl.Learner(X, y, clf)
    learner.run()

def test_WAE_rejuvenation():
    X, y = sl.utils.load_arff('toyset.arff')
    clf = sl.ensembles.WAE(rejuvenation_power=.5)
    learner = sl.Learner(X, y, clf)
    learner.run()

def test_REA():
    X, y = sl.utils.load_arff('toyset.arff')
    clf = sl.ensembles.REA()
    learner = sl.Learner(X, y, clf)
    learner.run()
