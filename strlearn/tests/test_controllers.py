"""Tests of implemented controllers."""
import sys
import strlearn as sl

sys.path.insert(0, '../..')


def test_bare_controller():
    """Test bare controller."""
    stream = sl.utils.ARFF('toyset.arff')
    ctrl = sl.controllers.Bare()
    learner = sl.learners.TestAndTrain(stream, controller=ctrl)
    learner.run()


def test_budget_controller():
    """Test budget controller."""
    stream = sl.utils.ARFF('toyset.arff')
    ctrl = sl.controllers.Budget()
    learner = sl.learners.TestAndTrain(stream, controller=ctrl)
    learner.run()


def test_BLALC_controller():
    """Test BLALC controller."""
    stream = sl.utils.ARFF('toyset.arff')
    ctrl = sl.controllers.BLALC()
    learner = sl.learners.TestAndTrain(stream, controller=ctrl)
    learner.run()
