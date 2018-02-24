#%load_ext autoreload
import sys
import os
sys.path.insert(0, '.')
import strlearn
reload(strlearn)
reload(strlearn.ensembles)
print strlearn.ensembles.__file__
from sklearn import naive_bayes
# %%

#
print "Hello"
nb_clf = naive_bayes.GaussianNB()
toystream = open('datasets/toyset.arff', 'r')
clf = strlearn.ensembles.WAE(
    base_classifier=nb_clf,
)
# %%

#
learner = strlearn.Learner(toystream, clf)
learner.run()
# %%
