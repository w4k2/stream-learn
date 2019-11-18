import strlearn as sl
import numpy as np

stream = sl.streams.StreamGenerator(n_chunks=5, chunk_size=500, weights=[0.2, 0.8], n_drifts=3, concept_sigmoid_spacing=5)
# clf = sl.ensembles.OOB(time_decay_factor=0.9)
#clf = sl.ensembles.UOB(time_decay_factor=0.9)
clf = sl.ensembles.OnlineBagging()
clf.set_base_clf(sl.ensembles.SampleWeightedMetaEstimator)
# clf = sl.ensembles.ChunkBasedEnsemble()
# clf = sl.classifiers.AccumulatedSamplesClassifier()
evaluator = sl.evaluators.TestThenTrainEvaluator()
evaluator.process(stream, clf)

print(evaluator.scores_)
print(np.mean(evaluator.scores_, axis=1))