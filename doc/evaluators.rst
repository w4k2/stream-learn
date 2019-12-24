#################
Stream Evaluators
#################

To estimate prediction measures in the context of data streams with
strict computational requirements and concept drifts, the ``evaluators`` module
of the ``stream-learn`` package implements two main estimation techniques
described in the literature in their batch-based versions.

Test-Then-Train Evaluator
=========================
.. :cite:`Gama2010`

The ``TestThenTrain`` class implements the `Test-Then-Train` evaluation procedure,
in which each individual data chunk is first used to test the classifier before
it is used for updating the existing model.

.. image:: plots/evaluators_ttt.png
    :width: 800px
    :align: center

The performance metrics returned by the evaluator are determined by the
``metrics`` parameter which accepts a tuple containing the functions of preferred quality
measures and can be specified during initialization.

Processing of the data stream is started by calling the ``process()`` function
which accepts two parameters (i.e. ``stream`` and ``clfs``) responsible for
defining the data stream and classifier, or a tuple of classifiers, employing
the ``partial_fit()`` function. The size of each data chunk is determined by
the ``chunk_size`` parameter from the ``StreamGenerator`` class. The results
of evaluation can be accessed using the ``scores`` attribute, which is a
three-dimensional array of shape (n_classifiers, n_chunks, n_metrics).


**Example -- single classifier**

.. code-block:: python

  from strlearn.evaluators import TestThenTrain
  from strlearn.ensembles import SEA
  from strlearn.utils.metrics import bac, f_score
  from strlearn.streams import StreamGenerator
  from sklearn.naive_bayes import GaussianNB

  stream = StreamGenerator(chunk_size=200, n_chunks=250)
  clf = SEA(base_estimator=GaussianNB())
  evaluator = TestThenTrain(metrics=(bac, f_score))

  evaluator.process(stream, clf)
  print(evaluator.scores)


**Example -- multiple classifiers**

.. code-block:: python

  from strlearn.evaluators import TestThenTrain
  from strlearn.ensembles import SEA
  from strlearn.utils.metrics import bac, f_score
  from strlearn.streams import StreamGenerator
  from sklearn.naive_bayes import GaussianNB
  from sklearn.tree import DecisionTreeClassifier

  stream = StreamGenerator(chunk_size=200, n_chunks=250)
  clf1 = SEA(base_estimator=GaussianNB())
  clf2 = SEA(base_estimator=DecisionTreeClassifier())
  clfs = (clf1, clf2)
  evaluator = TestThenTrain(metrics=(bac, f_score))

  evaluator.process(stream, clfs)
  print(evaluator.scores)

Prequential Evaluator
=====================
.. :cite:`Gama2013`

The `Prequential` procedure of assessing the predictive performance of stream
learning algorithms is implemented by the ``Prequential`` class. This estimation
technique is based on a forgetting mechanism in the form of a sliding window
instead of a separate data chunks. Window moves by a fixed number of instances
determined by the ``interval`` parameter for the ``process()`` function. After each
step, samples that are currently in the window are used to test the classifier
and then for updating the model.

.. image:: plots/evaluators_pr.png
    :width: 800px
    :align: center

Similar to the ``TestThenTrain`` evaluator, the object of the ``Prequential``
class can be initialized with a ``metrics`` parameter containing metrics names
and the size of the sliding window is equal to the ``chunk_size`` parameter from
the instance of ``StreamGenerator`` class.

**Example -- single classifer**

.. code-block:: python

  from strlearn.evaluators import Prequential
  from strlearn.ensembles import SEA
  from strlearn.utils.metrics import bac, f_score
  from strlearn.streams import StreamGenerator
  from sklearn.naive_bayes import GaussianNB

  stream = StreamGenerator()
  clf = SEA(base_estimator=GaussianNB())
  evaluator = TestThenTrain(metrics=(bac, f_score))

  evaluator.process(stream, clf, interval=100)
  print(evaluator.scores)


**Example -- multiple classifiers**

.. code-block:: python

  from strlearn.evaluators import Prequential
  from strlearn.ensembles import SEA
  from strlearn.utils.metrics import bac, f_score
  from strlearn.streams import StreamGenerator
  from sklearn.naive_bayes import GaussianNB
  from sklearn.tree import DecisionTreeClassifier

  stream = StreamGenerator(chunk_size=200, n_chunks=250)
  clf1 = SEA(base_estimator=GaussianNB())
  clf2 = SEA(base_estimator=DecisionTreeClassifier())
  clfs = (clf1, clf2)
  evaluator = Prequential(metrics=(bac, f_score))

  evaluator.process(stream, clfs, interval=100)
  print(evaluator.scores)

Metrics
=======
To improve the computational performance of presented evaluators, the
``stream-learn`` package uses its own implementations of metrics for classification
of imbalanced binary problems, which can be found in the ``utils.metrics`` module.
All implemented metrics are based on the confusion matrix.

.. image:: plots/confusion_matrix.png
    :align: center

Recall
------
.. :cite:`Powers2011`

Recall (also known as sensitivity or true positive rate) represents the
classifier's ability to find all the positive data samples in the dataset
(e.g. the minority class instances) and is denoted as

.. math::
   Recall = \frac{tp}{tp + fn}

**Example**

.. code-block:: python

  from strlearn.utils.metrics import recall

Precision
---------
.. :cite:`Powers2011`

Precision (also called positive predictive value) expresses the probability
of correct detection of positive samples and is denoted as

.. math::
   Precision = \frac{tp}{tp + fp}

**Example**

.. code-block:: python

  from strlearn.utils.metrics import precision

F-beta score
------------
.. :cite:`BaezaYates1999`

The F-beta score can be interpreted as a weighted harmonic mean of precision and
recall taking both metrics into account and punishing extreme values. The ``beta`` parameter determines the recall's weight. ``beta`` < 1 gives more weight to precision, while ``beta`` > 1 prefers recall.
The formula for the F-beta score is

.. math::
   F_\beta = (1+\beta^2) * \frac{Precision * Recall}{(\beta^2 * Precision) + Recall}

**Example**

.. code-block:: python

  from strlearn.utils.metrics import fbeta_score

F1 score
--------
.. :cite:`Sasaki2007`

The F1 score can be interpreted as a F-beta score, where :math:`\beta` parameter equals 1. It is a harmonic mean of precision and recall.
The formula for the F1 score is

.. math::
   F_1 = 2 * \frac{Precision * Recall}{Precision + Recall}

**Example**

.. code-block:: python

  from strlearn.utils.metrics import f1_score

Balanced accuracy (BAC)
-----------------------
.. :cite:`Brodersen2010,Kelleher2015`

The balanced accuracy for the multiclass problems is defined as the average of
recall obtained on each class. For binary problems it is denoted by the average
of recall and specificity (also called true negative rate).

.. math::
    Specificity = \frac{tn}{tn + fp}
.. math::
    BAC = \frac{Recall + Specificity}{2}

**Example**

.. code-block:: python

  from strlearn.utils.metrics import bac

Geometric mean score 1 (G-mean1)
--------------------------------
.. :cite:`Barandela2003,Kubat1997`

The geometric mean (G-mean) tries to maximize the accuracy on each of the
classes while keeping these accuracies balanced. For N-class problems it is
a N root of the product of class-wise recall. For binary classification
G-mean is denoted as the squared root of the product of the recall and specificity.

.. math::
    Gmean1 = \sqrt{Recall * Specificity}

**Example**

.. code-block:: python

  from strlearn.utils.metrics import geometric_mean_score_1

Geometric mean score 2 (G-mean2)
--------------------------------

The alternative definition of G-mean measure. For binary classification
G-mean is denoted as the squared root of the product of the recall and precision.

.. math::
    Gmean2 = \sqrt{Recall * Precision}

**Example**

.. code-block:: python

  from strlearn.utils.metrics import geometric_mean_score_2


References
----------
.. bibliography:: ../references_evaluators.bib
  :list: enumerated
  :all:
