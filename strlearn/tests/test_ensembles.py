"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")


def test_ACS():
    stream = sl.streams.StreamGenerator()
    clf = sl.ensembles.ChunkBasedEnsemble()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)
