#################
Stream Evaluators
#################

To estimate prediction measures in the context of data streams with
strict computational requirements and concept drifts, the ``evaluators`` module
of the ``stream-learn`` package implements two main estimation techniques
described in the literature in their block-based versions.

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
``metrics`` parameter which accepts a tuple containing the names of preferred
measures and can be specified during initialization.

Processing of the data stream is started by calling the ``process()`` function
which accepts two parameters (i.e. ``stream`` and ``clfs``) responsible for
defining the data stream and classifier, or a tuple of classifiers, employing
the ``partial_fit()`` function. The size of each data chunk is determined by
the ``chunk_size`` parameter from the ``StreamGenerator`` class. The results
of evaluation can be accessed using the ``scores`` attribute, which is a
three-dimensional array of shape (n_classifiers, n_chunks, n_metrics).


**Example**

.. code-block:: python

  from strlearn.evaluators import TestThenTrain
  from strlearn.ensembles import ChunkBasedEnsemble
  from strlearn.utils.metrics import bac, f_score
  from strlearn.streams import StreamGenerator
  from sklearn.naive_bayes import GaussianNB

  stream = StreamGenerator(chunk_size=200, n_chunks=250)
  clf = ChunkBasedEnsemble(base_estimator=GaussianNB())
  evaluator = TestThenTrain(metrics=(bac, f_score))

  evaluator.process(stream, clf)
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

**Example**

.. code-block:: python

  from strlearn.evaluators import Prequential
  from strlearn.ensembles import ChunkBasedEnsemble
  from strlearn.utils.metrics import bac, f_score
  from strlearn.streams import StreamGenerator
  from sklearn.naive_bayes import GaussianNB

  stream = StreamGenerator()
  clf = ChunkBasedEnsemble(base_estimator=GaussianNB())
  evaluator = TestThenTrain(metrics=(bac, f_score))

  evaluator.process(stream, clf, interval=100)
  print(evaluator.scores)

Metrics
=======
To improve the computational performance of presented evaluators, the
``stream-learn`` package uses its own implementations of metrics for classification
of imbalanced binary problems, which can be found in the ``utils.metrics`` module.
All implemented metrics are based on the confusion matrix.

.. image:: plots/confusion_matrix.png
    :align: center

Recall / True positive rate
---------------------------
.. :cite:`Powers2011`

.. code-block:: python

  from strlearn.utils.metrics import recall

.. math::
   Recall = \frac{tp}{tp + fn}

Precision / Positive predictive value
-------------------------------------
.. :cite:`Powers2011`

.. code-block:: python

  from strlearn.utils.metrics import precision

.. math::
   Precision = \frac{tp}{tp + fp}

F1 score
--------
.. :cite:`Sasaki2007`

.. code-block:: python

  from strlearn.utils.metrics import f_score

.. math::
   F1 = 2 * \frac{Precision * Recall}{Precision + Recall}

Balanced accuracy (BAC)
-----------------------
.. :cite:`Brodersen2010,Kelleher2015`

.. code-block:: python

  from strlearn.utils.metrics import bac

.. math::
    Specificity = \frac{tn}{tn + fp}
.. math::
    BAC = \frac{Recall * Specificity}{2}

Geometric mean score (G-mean)
-----------------------------
.. :cite:`Barandela2003,Kubat1997`

.. code-block:: python

  from strlearn.utils.metrics import geometric_mean_score

.. math::
    Gmean = \sqrt{Recall * Specificity}

References
----------
.. bibliography:: ../references_evaluators.bib
  :list: enumerated
  :all:
