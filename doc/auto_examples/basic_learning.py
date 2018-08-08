# -*- coding: utf-8 -*-
"""
==========================
Usage of the WAE algorithm
==========================
This example shows a basic stream processing using WAE algorithm.

"""

# Authors: Paweł Ksieniewicz <pawel.ksieniewicz@pwr.edu.pl>
# License: MIT

from sklearn import naive_bayes
from strlearn import Learner, ensembles
import arff
import numpy as np


###############################################################################
# Preparing data for learning
###############################################################################

###############################################################################
# lorem ipsum
with open('datasets/toyset.arff', 'r') as stream:
    dataset = arff.load(stream)
    data = np.array(dataset['data'])
    X = data[:, :-1].astype(np.float)
    y = data[:, -1]

base_classifier = naive_bayes.GaussianNB()

###############################################################################
# lorem ipsum

clf = ensembles.WAE(
    ensemble_size=10, theta=.1,
    post_pruning=False, pruning_criterion='diversity',
    weight_calculation_method='kuncheva',
    aging_method='weights_proportional',
    rejuvenation_power=.5
)
clf.set_base_clf(base_classifier)

###############################################################################
# lorem ipsum

learner = Learner(X, y, clf)
learner.run()
