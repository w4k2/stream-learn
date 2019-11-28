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
from sklearn.naive_bayes import GaussianNB

stream = sl.streams.StreamGenerator(n_chunks=250, n_drifts=1)
clf = GaussianNB()
evaluator = sl.evaluators.TestThenTrainEvaluator()

evaluator.process(stream, clf)

print(evaluator.scores_)
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
