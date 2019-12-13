#! /usr/bin/env python
"""Toolbox for streaming data."""
from __future__ import absolute_import

import codecs
import os

from setuptools import find_packages, setup

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get __version__ from _version.py
ver_file = os.path.join("strlearn", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "stream-learn"
DESCRIPTION = "The stream-learn module is a set of tools necessary for processing data streams using scikit-learn estimators."
MAINTAINER = "P. Ksieniewicz"
MAINTAINER_EMAIL = "pawel.ksieniewicz@pwr.edu.pl"
URL = "https://w4k2.github.io/stream-learn/"
LICENSE = "GPL-3.0"
DOWNLOAD_URL = "https://github.com/w4k2/stream-learn"
VERSION = __version__
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "scikit-learn"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]


setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
)
