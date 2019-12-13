################
Classifiers
################

In addition, the ``stream-learn`` library also implements a simple single
classifier model implementing the ``partial_fit()`` method and a Meta estimator
adapted to be used with some of the ensemble methods found in the ``ensembles``
module. Those two models can be found in the ``classifiers`` module.

Accumulated Samples Classifier
------------------------------

The ``AccumulatedSamplesClassifier`` class takes the base classifier as a
``base_clf`` parameter during initialization and extends the given model with
the ``partial_fit()`` function adapted for data streams classification.
This function concatenates observed data chunks, and in each step fits
the model on all samples encountered so far.

**Example**

.. code-block:: python

  from strlearn.evaluators import TestThenTrain
  from strlearn.streams import StreamGenerator
  from strlearn.classifiers import AccumulatedSamplesClassifier

  from sklearn.naive_bayes import GaussianNB

  stream = StreamGenerator()
  clf = AccumulatedSamplesClassifier(base_clf=GaussianNB())
  evaluator = TestThenTrain()

  evaluator.process(stream, clf)
  print(evaluator.scores)

Sample-Weighted Meta Estimator
------------------------------

The ``SampleWeightedMetaEstimator`` class implements a meta estimator designed
to allow the use of a wider range of classification models as base classifiers
in ensemble methods based on `online bagging`. It extends the ``partial_fit()``
method of a given model by an additional  ``sample_weight`` parameter which
allows for using classifiers such as ``MLPClassifier`` from ``scikit-learn``
package as base models for ``OnlineBagging``, ``OOB`` and ``UOB`` from
``ensembles`` module.

**Example**

.. code-block:: python

    from strlearn.evaluators import TestThenTrain
    from strlearn.streams import StreamGenerator
    from strlearn.classifiers import SampleWeightedMetaEstimator
    from strlearn.ensembles import OOB

    from sklearn.neural_network import MLPClassifier


    stream = StreamGenerator(n_chunks=10)
    base = SampleWeightedMetaEstimator(base_classifier=MLPClassifier())
    clf = OOB(base_estimator=base, n_estimators=2)
    evaluator = TestThenTrain()

    evaluator.process(stream, clf)
    print(evaluator.scores)
