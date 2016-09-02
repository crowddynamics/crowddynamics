#!/usr/bin/env bash

# Export path to minicond3
export PATH=~/miniconda3:$PATH
export PYTHONPATH=~/miniconda3:$PYTHONPATH

# Create conda environment for crowd dynamics
conda create --name crowd34 python=3.4
source activate crowd34

conda install numpy numba pyqt pyqtgraph h5py pandas scipy
pip install scikit-fmm
pip install ruamel.yaml
