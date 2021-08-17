from setuptools import setup, find_packages

setup(name='autoinit',
    version='0.0.1',
    description='AutoInit: Analytic Mean- and Variance-Preserving Weight Initialization for Neural Networks',
    author='@garrettbingham',
    author_email='garrett.bingham@cognizant.com',
    packages=find_packages(include=['autoinit', 'autoinit.*']),
)
