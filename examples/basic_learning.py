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


###############################################################################
# Preparing data for learning
###############################################################################

###############################################################################
# lorem ipsum

stream = open('datasets/toyset.arff', 'r')
base_classifier = naive_bayes.GaussianNB()

###############################################################################
# lorem ipsum

clf = ensembles.WAE(
    base_classifier=base_classifier,
    ensemble_size=10,
    theta=.1,
    is_post_pruning=False, pruning_criterion='diversity',
    weight_calculation_method='kuncheva',
    aging_method='weights_proportional',
    rejuvenation_power=.5
)

###############################################################################
# lorem ipsum

learner = Learner(stream, clf)
learner.run()
