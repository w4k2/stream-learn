# stream-learn

[![Travis Status](https://travis-ci.org/w4k2/stream-learn.svg?branch=master)](https://travis-ci.org/w4k2/stream-learn)
[![Coverage Status](https://coveralls.io/repos/github/w4k2/stream-learn/badge.svg?branch=master)](https://coveralls.io/github/w4k2/stream-learn?branch=master)
[![CircleCI Status](https://circleci.com/gh/w4k2/stream-learn.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/w4k2/stream-learn/tree/master)

stream-learn is a Python package equipped with a procedures to process data streams using estimators with API compatible with scikit-learn.

## Documentation

API documentation with set of examples may be found on the [documentation page](https://w4k2.github.io/stream-learn/).

## Installation

stream-learn is available on the PyPi and you may install it with pip:

```
pip install stream-learn
```

## Example usage

```python
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Initialize list of scikit-learn classifiers with partial_fit() function
clf = [MLPClassifier(), GaussianNB()]

# Declare data stream
stream = sl.streams.StreamGenerator(n_chunks=10, n_drifts=1)

# Select vector of metrics
metrics = [sl.utils.metrics.bac, sl.utils.metrics.f_score]

# Initialize evaluator with given metrics
evaluator = sl.evaluators.TestThenTrain(metrics)

# Run evaluator over stream with classifier
evaluator.process(stream, clf)
```

```python
>>> print(evaluator.scores)
[[[0.29730274 0.29145729]
  [0.34494021 0.36097561]
  [0.43464118 0.44878049]
  [0.42579578 0.36666667]
  [0.45569557 0.4171123 ]
  [0.47020869 0.44791667]
  [0.4645207  0.46534653]
  [0.525      0.5177665 ]
  [0.4893617  0.46875   ]]

 [[0.87701288 0.88038278]
  [0.90091448 0.9047619 ]
  [0.89930938 0.9047619 ]
  [0.85376189 0.82681564]
  [0.61521152 0.60913706]
  [0.64714185 0.61538462]
  [0.64556129 0.62564103]
  [0.74       0.74      ]
  [0.80820955 0.80597015]]]
```

<!--

### About

If you use stream-learn in a scientific publication, we would appreciate citations to the following paper:

```
@article{key:key,
author  = {abc},
title   = {def},
journal = {ghi},
year    = {2018},
volume  = {1},
number  = {1},
pages   = {1-5},
url     = {http://jkl}
}
```
-->
