import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
import numpy as np

# stream = sl.streams.StreamGenerator(n_chunks=5, chunk_size=500, weights=[0.2, 0.8], n_drifts=3, concept_sigmoid_spacing=999)
# clf1 = sl.ensembles.OOB(time_decay_factor=0.9)
# clf2 = sl.ensembles.UOB(time_decay_factor=0.9)
# clf3 = sl.ensembles.OnlineBagging()
# clf.set_base_clf(sl.ensembles.SampleWeightedMetaEstimator)
##clf4 = sl.ensembles.ChunkBasedEnsemble()
# clf5 = sl.classifiers.AccumulatedSamplesClassifier()

# clf = {clf1, clf2, clf3, clf4, clf5}

clf = [GaussianNB(), GaussianNB()]

n_chunks = 5
a = []
b = []
for i in range(1000):
    stream = sl.streams.StreamGenerator(
        n_chunks=n_chunks,
        chunk_size=500,
        weights=[0.2, 0.8],
        n_drifts=3,
        concept_sigmoid_spacing=999,
    )
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, [clone(c) for c in clf])

    print("\n# TTT\n# %04i" % i, np.mean(evaluator.scores_, axis=1), "\n")

    a.append(np.mean(evaluator.scores_, axis=1))

    stream = sl.streams.StreamGenerator(
        n_chunks=n_chunks,
        chunk_size=500,
        weights=[0.2, 0.8],
        n_drifts=3,
        concept_sigmoid_spacing=999,
    )
    evaluator = sl.evaluators.PrequentialEvaluator()
    evaluator.process(stream, [clone(c) for c in clf], interval=500)

    print("\n# Preq\n# %04i" % i, np.mean(evaluator.scores_, axis=1))

    b.append(np.mean(evaluator.scores_, axis=1))
    exit()

a = np.array(a)

a = np.std(a, axis=0)

print(a)
