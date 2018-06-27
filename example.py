import warnings
import strlearn as sl

warnings.simplefilter('ignore', DeprecationWarning)

X, y = sl.utils.load_arff('../arff/HyperplaneFaster.arff')
ctrl = sl.controllers.Bare()

learner = sl.Learner(X, y, controller=ctrl)
learner.run()

print(learner.scores)
