#!/bin/bash

# Create conda environment for crowd dynamics
# conda config --add channels conda-forge

# Environment name will be crowd34
conda env create -f environment.yaml
source activate crowd34
