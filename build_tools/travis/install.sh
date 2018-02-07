#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone pydicom/pydicom repository in to a local repository.

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

# At the time of writing numpy 1.9.1 is included in the travis
# virtualenv but we want to use the numpy installed through apt-get
# install.
deactivate
# Create a new virtualenv using system site packages for python, numpy
virtualenv --system-site-packages testvenv
source testvenv/bin/activate
pip install scikit-learn pandas nose nose-timer pytest pytest-cov codecov \
    sphinx numpydoc enum liac-arff

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

python setup.py develop
ccache --show-stats
