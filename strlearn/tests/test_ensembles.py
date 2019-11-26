"""Ensemble tests."""

import sys
import strlearn as sl

sys.path.insert(0, "../..")


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10)


def test_CBE():
    "Bare CBE"
    stream = get_stream()
    clf = sl.ensembles.ChunkBasedEnsemble()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_OOB():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.OOB()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_OB():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.OnlineBagging()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_UOB():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.UOB()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_pp_WAE():
    """Post pruned WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(post_pruning=True, ensemble_size=5)
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE_wcm1():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(weight_calculation_method="same_for_each")
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE_wcm2():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(weight_calculation_method="proportional_to_accuracy")
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE_wcm3():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(weight_calculation_method="pta_related_to_whole")
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE_wcm4():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(weight_calculation_method="bell_curve")
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE_am2():
    """Various aging methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(aging_method="constant")
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE_am3():
    """Various aging methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(aging_method="gaussian")
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_WAE_rejuvenation():
    """Rejuvenation of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(rejuvenation_power=0.5)
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_pp_WAE_rejuvenation():
    """Post pruning with rejuvenation WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(rejuvenation_power=0.5, post_pruning=True)
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)
