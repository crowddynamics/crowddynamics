#!/bin/bash

# Create conda environment for crowd dynamics
# conda config --add channels conda-forge
conda env create -f environment.yaml --name crowd35
source activate crowd35
