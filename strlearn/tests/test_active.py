"""Testing active learning approach."""
import sys
import strlearn as sl

sys.path.insert(0, '../..')


def test_active():
    """Test active learning using BLALC."""
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

    for controller in controllers:
            stream = sl.utils.ARFF('toyset.arff')
            learner = sl.learners.TestAndTrain(stream,
                                               chunk_size=500,
                                               controller=controller)
            learner.run()
