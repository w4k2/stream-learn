"""
===================================================================
The simplest experiment example with one classifier and two metrics
===================================================================

Lorem impsum.

"""

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from strlearn.evaluators import TestThenTrain
from strlearn.metrics import recall
from strlearn.streams import StreamGenerator

clf = GaussianNB()


stream = StreamGenerator(n_chunks=30, n_drifts=1)



metrics = [accuracy_score, recall]



evaluator = TestThenTrain(metrics)


evaluator.process(stream, clf)

print(evaluator.scores.shape)
evaluator.scores

##############################################################################
# Processing
##############################################################################

##############################################################################
# Olaboga, jakie ważne bardzo.


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
