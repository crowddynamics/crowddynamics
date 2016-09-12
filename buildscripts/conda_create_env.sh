#!/usr/bin/env bash

# Create conda environment for crowd dynamics
conda create --name crowd35 python=3.5
source activate crowd35

conda install yaml numpy numba matplotlib pyqt pyqtgraph h5py pandas scipy scikit-image shapely

pip install ruamel.yaml
pip install scikit-fmm
