######################################
Welcome to stream-learn documentation!
######################################

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
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

.. image:: _static/disco.png

The ``stream-learn`` module is a set of tools necessary for processing data streams using ``scikit-learn`` estimators. The batch processing approach is used here, where the dataset is passed to the classifier in smaller, consecutive subsets called `chunks`. The module consists of five sub-modules:

- `streams <streams.html>`_ - containing a data stream generator that allows obtaining both stationary and dynamic distributions in accordance with various types of concept drift (also in the field of a priori probability, i.e. dynamically unbalanced data) and a parser of the standard ARFF file format.
- `evaluators <evaluators.html>`_ - containing classes for running experiments on stream data in accordance with the Test-Then-Train and Prequential methodology.
- `classifiers <classifiers.html>`_ - containing sample stream classifiers,
- `ensembles <ensembles.html>`_ - containing standard team models of stream data classification,
- utils - containing typical classification quality metrics in data streams.

You can read more about each module in the User Guide.

`Getting started <install.html>`_
---------------------------------

Information to install and usage of the package.

`API Documentation <api.html>`_
-------------------------------

The exact API of all functions and classes, as given in the
docstring. The API documents expected types and allowed features for
all functions, and all parameters available for the algorithms.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples illustrating the use of the different algorithms.

See the `README <https://github.com/w4k2/stream-learn/blob/master/README.md>`_
for more information.
