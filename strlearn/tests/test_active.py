import sys
import strlearn as sl

sys.path.insert(0, '../..')

"""
def test_active():
    tresholds = [.3, .7]
    budgets = [.3, .7]

    controllers = []

    controllers.append(sl.controllers.Bare())

    for budget in budgets:
        controllers.append(sl.controllers.Budget(budget=budget))

    for budget in budgets:
        for treshold in tresholds:
            controllers.append(sl.controllers.BLALC(
                budget=budget, treshold=treshold))

    X, y = sl.utils.load_arff('toyset.arff')

    for controller in controllers:
            learner = sl.Learner(X, y,
                                 evaluate_interval=1000, chunk_size=500,
                                 controller=controller)
            learner.run()
"""
