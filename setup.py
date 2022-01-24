
# Copyright (C) 2021-2022 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# AutoInit Software in commercial settings.
#
# END COPYRIGHT
from setuptools import setup, find_packages

setup(name='autoinit',
    version='0.0.4',
    description='AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks',
    author='@garrettbingham',
    author_email='garrett.bingham@cognizant.com',
    packages=find_packages(include=['autoinit', 'autoinit.*']),
)
