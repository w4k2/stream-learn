import sys
sys.path.insert(0, '../..')

import strlearn

from sklearn import tree

toystream = open('datasets/toyset.arff', 'r')
dt_clf = tree.DecisionTreeClassifier()

def test_WAE():
    clf = strlearn.ensembles.WAE(base_classifier = dt_clf)
    learner = strlearn.Learner(toystream, clf, verbose=False)
    learner.run()
    #learner.serialize('result.csv')
