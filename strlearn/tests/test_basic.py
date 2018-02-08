import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn import tree
import WAE
import streamLearner

dbname = 'toyset'

def test_something():
    assert 5 == 5

def test_WAE():
    base_clf = tree.DecisionTreeClassifier()
    clf = WAE.WAE(base_classifier = base_clf)
    stream = open('datasets/%s.arff' % dbname, 'r')
    learner = streamLearner.StreamLearner(stream, clf, verbose=False)
    learner.run()
    learner.serialize('result.csv')
