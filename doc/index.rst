######################################
Welcome to stream-learn documentation!
######################################

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User guide

   streams
   evaluators
   ensembles
   classifiers

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Other informations

   about


User Guide
----------

.. image:: _static/disco.png

The ``stream-learn`` module is a set of tools necessary for processing data streams using ``scikit-learn`` estimators. The batch processing approach is used here, where the dataset is passed to the classifier in smaller, consecutive subsets called `chunks`. The module consists of five sub-modules:

- `streams <streams.html>`_ - containing a data stream generator that allows obtaining both stationary and dynamic distributions in accordance with various types of concept drift (also in the field of a priori probability, i.e. dynamically unbalanced data) and a parser of the standard ARFF file format.
- `evaluators <evaluators.html>`_ - containing classes for running experiments on stream data in accordance with the Test-Then-Train and Prequential methodology.
- `classifiers <classifiers.html>`_ - containing sample stream classifiers,
- `ensembles <ensembles.html>`_ - containing standard team models of stream data classification,
- `metrics <evaluators.html>`_ - containing typical classification quality metrics in data streams.

You can read more about each module in the User Guide.

`Getting started <quickstart.html>`_
------------------------------------

A brief description of the installation process and basic usage of the module in a simple experiment.

`API Documentation <api.html>`_
-------------------------------

Precise API description of all the classes and functions implemented in the module.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples illustrating the use of all module elements.

See the `README <https://github.com/w4k2/stream-learn/blob/master/README.md>`_
for more information.
