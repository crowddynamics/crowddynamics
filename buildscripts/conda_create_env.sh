#!/usr/bin/env bash

# Create conda environment for crowd dynamics
conda create --name crowd34 python=3.4
source activate crowd34

conda install yaml numpy numba pyqt pyqtgraph h5py pandas scipy scikit-image shapely
pip install scikit-fmm
pip install ruamel.yaml
