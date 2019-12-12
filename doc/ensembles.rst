####################
Classifier Ensembles
####################

An ensemble (also known as multiple classifier system or committee) consists of
a set of base classifiers whose predictions are combined to label new instances.
Combining classifiers have been proved to be an effective way of dividing
complex learning problems into sub-problems as well as improving predictive
accuracy. A well-tuned ensemble should contain both strong and diverse base
models.

**Classifier ensemble diagram**

.. image:: plots/ensemble.png
    :width: 650 px
    :align: center

Under the data stream scenario, based on the way of examples processing, the
ensembles can be categorized as `chunk-based` or `online`. The ``stream-learn``
package implements various ensemble methods for data stream classification,
which can be found in the ``ensembles`` module.


Chunk-based Ensembles for Data Streams
======================================

Chunk-based approaches process successively incoming data chunks containing a
predetermined number of instances. The learning algorithm can repeatedly process
training samples located in a given data chunk to learn base models. It is worth
noting that this does not mean that batch processing can only be used when new
instances arrive in chunks. These approaches can also be used when instances
arrive individually, if we store each new sample in a buffer until its size is
equal to the size of the chunk.

Chunk-Based Ensemble
--------------------


Weighted Aging Ensemble (WAE)
-----------------------------


Online Ensembles for Data Streams
=================================

Online approaches, unlike those based on batch processing, process each new
sample separately. These methods have been developed for applications with
memory and computational limitations (i.e. where the amount of incoming data is
extensive). Online methods can also be used in cases where data samples do not
arrive separately. These types of methods can process each instances of data
chunk are individually and can therefore be used in an environment where data
arrives in batches.



Online Bagging (OB)
-------------------


Oversamping-Based Online Bagging (OOB)
--------------------------------------


Undersampling-Based Online Bagging (UOB)
----------------------------------------


References
----------
.. bibliography:: ../references_ensembles.bib
  :list: enumerated
  :all:
