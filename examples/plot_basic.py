"""
=======================================================
Basic example of stationary synthetic stream processing
=======================================================

Lorem impsum.

"""


import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import strlearn as sl

# Initialize list of scikit-learn classifiers with partial_fit() function
clfs = [MLPClassifier(), GaussianNB()]

# Declare data stream
stream = sl.streams.StreamGenerator(n_chunks=30, n_drifts=1)

# Select vector of metrics
metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.f1_score]

# Initialize evaluator with given metrics
evaluator = sl.evaluators.TestThenTrain(metrics)


##############################################################################
# Processing
##############################################################################

##############################################################################
# Olaboga, jakie ważne bardzo.

# Run evaluator over stream with classifier
evaluator.process(stream, clfs)

evaluator.scores

##############################################################################
# Plotting
##############################################################################


fig, ax = plt.subplots(1, len(metrics), figsize=(12, 4))
labels = ["MLP", "GNB"]
for m, metric in enumerate(metrics):
    ax[m].set_title(metric.__name__)
    ax[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[m, :, i], label=labels[i])
    ax[m].legend()
