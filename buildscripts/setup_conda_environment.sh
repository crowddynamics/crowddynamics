#!/bin/bash

set -v

CONDA_INSTALL="conda install -q -y"
PIP_INSTALL="pip install -q"

# Deactivate any environment
source deactivate
# Display root environment (for debugging)
conda list
# Clean up any left-over from a previous build
# (note workaround for https://github.com/conda/conda/issues/2679:
#  `conda env remove` issue)
conda remove --all -q -y -n $CONDA_ENV
# Scipy, CFFI, jinja2 and IPython are optional dependencies, but exercised in the test suite
conda env create -f environments/$CONDA_ENV.yml -q -y
