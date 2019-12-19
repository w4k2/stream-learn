"""
===================================================================
The simplest experiment example with one classifier and two metrics
===================================================================

Lorem impsum.

"""

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

from strlearn.streams import StreamGenerator

stream = StreamGenerator(n_chunks=30, n_drifts=1)


from sklearn.metrics import accuracy_score
from strlearn.metrics import recall

metrics = [accuracy_score, recall]


from strlearn.evaluators import TestThenTrain

evaluator = TestThenTrain(metrics)


evaluator.process(stream, clf)

print(evaluator.scores.shape)
evaluator.scores

##############################################################################
# Processing
##############################################################################

##############################################################################
# Olaboga, jakie ważne bardzo.

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3), dpi=400)

for m, metric in enumerate(metrics):
    plt.plot(evaluator.scores[0, :, m], label=metric.__name__)

plt.title("Basic example of stream processing")
plt.ylim(0, 1)
plt.ylabel("Quality")
plt.xlabel("Chunk")

plt.legend()
plt.tight_layout()
plt.savefig("simplest.png", transparent=True)
