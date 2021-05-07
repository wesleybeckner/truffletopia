from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = ''  # use '' for first of series, number for 1 and above
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "Truffletopia"
# Long description will go up on the pypi page
long_description = """
License
=======
``truffletopia`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2021--, Wesley Beckner, The University of Washington.
"""

NAME = "truffletopia" #YOUR PACKAGE NAME
MAINTAINER = "Wesley Beckner" #YOUR NAME
MAINTAINER_EMAIL = "wesleybeckner@gmail.com" #EMAIL OR DUMMY EMAIL
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/wesleybeckner/truffletopia" #GITHUB URL
DOWNLOAD_URL = ""
LICENSE = "MIT" #CHOSEN LICENSE
AUTHOR = "Wesley Beckner" #YOUR NAME
AUTHOR_EMAIL = "wesleybeckner@gmail.com" #EMAIL OR DUMMY EMAIL
PLATFORMS = "OS Independent" #OPERATING SYSTEM
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'truffletopia': [pjoin('data', '*')]}
REQUIRES = ["numpy", "pandas", "scipy", "sklearn"]
