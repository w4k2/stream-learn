import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn import tree

from Learner import Learner
from WAE import WAE

toystream = open('datasets/toyset.arff', 'r')
dt_clf = tree.DecisionTreeClassifier()

def test_WAE():
    clf = WAE(base_classifier = dt_clf)
    learner = Learner(toystream, clf, verbose=False)
    learner.run()
    #learner.serialize('result.csv')
