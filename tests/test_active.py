"""Testing active learning approach."""
import sys
from .context import strlearn as sl


def test_active():
    """Test active learning using BLALC."""
    tresholds = [0.3, 0.7]
    budgets = [0.3, 0.7]

    controllers = []

    controllers.append(sl.controllers.Bare())

    for budget in budgets:
        controllers.append(sl.controllers.Budget(budget=budget))

    for budget in budgets:
        for treshold in tresholds:
            controllers.append(sl.controllers.BLALC(budget=budget, treshold=treshold))

    stream = sl.utils.StreamGenerator(
        drift_type="sudden", n_chunks=10, n_drifts=1, n_features=4, chunk_size=100
    )
    for controller in controllers:
        learner = sl.learners.TestAndTrain(stream, controller=controller)
        learner.run()
        stream.reset()
