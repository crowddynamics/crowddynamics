#!/usr/bin/env bash
# Environment variables:
#   CONDA_ENV: Name of the conda environment
#   PYTHON: Version of python to use

# Install Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda

# The next couple lines fix a crash with multiprocessing on Travis and are not specific to using Miniconda
sudo rm -rf /dev/shm
sudo ln -s /run/shm /dev/shm

# Install conda environment from file
conda env create \
      --quiet \
      --yes \
      -n $CONDA_ENV \
      python=$PYTHON \
      -f environment-dev.yml

source activate $CONDA_ENV

# Install crowddynamics
python setup.py install
