"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")


def test_ACS():
    stream = sl.generators.DriftedStream(sigmoid_spacing=999)
    clf = sl.ensembles.ChunkBasedEnsemble()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)