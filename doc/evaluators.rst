#################
Stream Evaluators
#################

Test-Then-Train Evaluator
=========================
:cite:`Gama2010`

.. image:: plots/evaluators_ttt.png
    :width: 800px
    :align: center

.. code-block:: python

  from strlearn.evaluators import TestThenTrain

Prequential Evaluator
=====================
:cite:`Gama2013`

.. image:: plots/evaluators_pr.png
    :width: 800px
    :align: center

.. code-block:: python

  from strlearn.evaluators import Prequential

Metrics
=======
To improve the computational performance of presented evaluators, the
``stream-learn`` package has its own implementation of metrics for classification
of imbalanced binary problems, which can be found in the ``utils.metrics`` module.
All implemented metrics are based on the confusion matrix.

.. image:: plots/confusion_matrix.png
    :align: center

Recall / True positive rate
---------------------------
:cite:`Powers2011`

.. code-block:: python

  from strlearn.utils.metrics import recall

.. math::
   Recall = \frac{tp}{tp + fn}

Precision / Positive predictive value
-------------------------------------
:cite:`Powers2011`

.. code-block:: python

  from strlearn.utils.metrics import precision

.. math::
   Precision = \frac{tp}{tp + fp}

F1 score
--------
:cite:`Sasaki2007`

.. code-block:: python

  from strlearn.utils.metrics import f_score

.. math::
   F1 = 2 * \frac{Precision * Recall}{Precision + Recall}

Balanced accuracy (BAC)
-----------------------
:cite:`Brodersen2010,Kelleher2015`

.. code-block:: python

  from strlearn.utils.metrics import bac

.. math::
    Specificity = \frac{tn}{tn + fp}
.. math::
    BAC = \frac{Recall * Specificity}{2}

Geometric mean score (G-mean)
-----------------------------
:cite:`Barandela2003,Kubat1997`

.. code-block:: python

  from strlearn.utils.metrics import geometric_mean_score

.. math::
    Gmean = \sqrt{Recall * Specificity}

References
----------
.. bibliography:: ../references.bib
    :list: enumerated
    :all:
