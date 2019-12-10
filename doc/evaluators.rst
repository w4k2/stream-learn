#################
Stream Evaluators
#################

Test-Then-Train Evaluator
=========================
.. image:: plots/evaluators_ttt.png
    :width: 800px
    :align: center

Prequential Evaluator
=====================
.. image:: plots/evaluators_pr.png
    :width: 800px
    :align: center

Metrics
=======
.. image:: plots/confusion_matrix.png
    :align: center

Recall / True positive rate
------

.. math::
   Recall = \frac{tp}{tp + fn}

Precision / Positive predictive value
---------

.. math::
   Precision = \frac{tp}{tp + fp}

F1 score
-------

.. math::
   F1 = 2 * \frac{Precision * Recall}{Precision + Recall}

Balanced accuracy (BAC)
-----------------------

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
