#################
Stream Evaluators
#################

Test-Then-Train Evaluator
=========================
:cite:`Gama2010`

.. image:: plots/evaluators_ttt.png
    :width: 800px
    :align: center

Prequential Evaluator
=====================
:cite:`Gama2013`

.. image:: plots/evaluators_pr.png
    :width: 800px
    :align: center

Metrics
=======
.. image:: plots/confusion_matrix.png
    :align: center

Recall / True positive rate
------
:cite:`Powers2011`

.. math::
   Recall = \frac{tp}{tp + fn}

Precision / Positive predictive value
---------
:cite:`Powers2011`

.. math::
   Precision = \frac{tp}{tp + fp}

F1 score
-------
:cite:`Sasaki2007`

.. math::
   F1 = 2 * \frac{Precision * Recall}{Precision + Recall}

Balanced accuracy (BAC)
-----------------------
:cite:`Brodersen2010,Kelleher2015`

.. math::
    Specificity = \frac{tn}{tn + fp}
.. math::
    BAC = \frac{Recall * Specificity}{2}

Geometric mean score (G-mean)
-----------------------------
:cite:`Barandela2003,Kubat1997`


.. math::
    Gmean = \sqrt{Recall * Specificity}

References
----------
.. bibliography:: ../references.bib
    :list: enumerated
    :all:
