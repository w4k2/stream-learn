#! /usr/bin/env python
"""Toolbox for streaming data."""
from __future__ import absolute_import

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('strlearn', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'stream-learn'
DESCRIPTION = 'Toolbox for streaming data.'
MAINTAINER = 'P. Ksieniewicz'
MAINTAINER_EMAIL = 'pawel.ksieniewicz@pwr.edu.pl'
URL = 'https://github.com/w4k2/stream-learn'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/w4k2/stream-learn'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Programming Language :: Python',
               'Topic :: Scientific/Engineering',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7']


setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES)
