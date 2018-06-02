# stream-learn

[![Travis Status](https://travis-ci.org/w4k2/stream-learn.svg?branch=master)](https://travis-ci.org/w4k2/stream-learn)
[![Coveralls Status](https://coveralls.io/repos/w4k2/stream-learn/badge.svg?branch=master&service=github)](https://coveralls.io/r/w4k2/stream-learn)
[![CircleCI Status](https://circleci.com/gh/w4k2/stream-learn.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/w4k2/stream-learn/tree/master)
[![KSSK](https://img.shields.io/badge/KSSK-alive-green.svg)](http://kssk.pwr.edu.pl)

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
import strlearn
from sklearn import neural_network

clf = neural_network.MLPClassifier()
X, y = strlearn.utils.load_arff('toyset.arff')
learner = strlearn.Learner(X, y, clf)
learner.run()
```

### About

If you use stream-learn in a scientific publication, we would appreciate citations to the following paper:

```
@article{JMLR:v18:16-365,
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
