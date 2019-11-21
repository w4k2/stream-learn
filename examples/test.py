import strlearn as sl
import numpy as np

stream = sl.streams.StreamGenerator(n_chunks=5, chunk_size=500, weights=[0.2, 0.8], n_drifts=3, concept_sigmoid_spacing=999)
clf1 = sl.ensembles.OOB(time_decay_factor=0.9)
clf2 = sl.ensembles.UOB(time_decay_factor=0.9)
clf3 = sl.ensembles.OnlineBagging()
# clf.set_base_clf(sl.ensembles.SampleWeightedMetaEstimator)
clf4 = sl.ensembles.ChunkBasedEnsemble()
clf5 = sl.classifiers.AccumulatedSamplesClassifier()

clf = {
    clf1,
    clf2,
    clf3,
    clf4,
    clf5
}

evaluator = sl.evaluators.PrequentialEvaluator()
# evaluator = sl.evaluators.TestThenTrainEvaluator()
evaluator.process(stream, clf, interval=500)

# print(evaluator.scores_.shape)
print(np.mean(evaluator.scores_, axis=1))