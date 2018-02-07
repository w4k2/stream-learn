from sklearn import tree
from strlearn import WAE
from strlearn import streamLearner

def test_something():
    assert 5 == 5

def test_WAE():
    base_clf = tree.DecisionTreeClassifier()
    dbname = 'RBFGradualRecurring'
    clf = WAE.WAE(base_classifier = base_clf)
    stream = open('datasets/%s.arff' % dbname, 'r')
    learner = streamLearner.StreamLearner(stream, clf, verbose=False)
