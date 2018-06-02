import sys
import strlearn as sl

sys.path.insert(0, '../..')

def test_bare_controller():
    X, y = sl.utils.load_arff('toyset.arff')
    ctrl = sl.controllers.Bare()

    learner = sl.Learner(X, y, controller=ctrl)
    learner.run()

def test_budget_controller():
    X, y = sl.utils.load_arff('toyset.arff')
    ctrl = sl.controllers.Budget()
    learner = sl.Learner(X, y, controller=ctrl)
    learner.run()


def test_BLALC_controller():
    X, y = sl.utils.load_arff('toyset.arff')
    ctrl = sl.controllers.BLALC()
    learner = sl.Learner(X, y, controller=ctrl)
    learner.run()
