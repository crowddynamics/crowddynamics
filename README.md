# Crowd Dynamics
Crowd dynamics is a simulation environment written in Python package for simulation movement of crowds. The project was created in summer 2016 for Systems Analysis Laboratory at Aalto University in Finland. The [documentation](http://jaantollander.github.io/crowddynamics/) has more detail about the project.


## Installation
Crowd dynamics is tested on Ubuntu 16.04 using Python 3.6.

Install the Conda package manager. The [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution is the easiest to install.

Clone the `crowddynamics` repository.
```bash
git clone https://github.com/jaantollander/crowddynamics.git
```

Inside the `crowddynamics` directory, install the Conda environment, activate it, and install `crowddynamics` as an editable.
```bash
conda env create -f environment.yml
conda activate crowd36
pip install --editable .
```

## Graphical User Interface
For the graphical user interface, we need to install the crowd dynamics QT gui. Start by cloning the repository. 
```bash
git clone https://github.com/jaantollander/crowddynamics-qtgui
```

Inside `crowddynamics-qtgui` directory, install the requirements and install `crowddynamics-qtgui` as editable. We must install `pyqt` version 4 using Conda from the `conda-force` channel for it to work correctly. Be sure that you activate the environment where you installed `crowddynamics`.  
```bash
conda activate crowd36
pip install -r requirements.txt
conda install pyqt=4 -c conda-force
pip install --editable .
```

## Usage
There are examples simulations inside the `crowddynamics/examples` directory.

We can run simulations from the commandline commandline. We can use the `--help` option to see all the commands.
```bash
crowddynamics --help
```
We can run simulations using the following command.
```bash
crowddynamics --simulation_cfg="crowddynamics/examples/simulations.py"
```

## Tests
Install test dependencies.
```bash
pip install -r requirements-tests.txt
```
Run `pystest` test suite.
```bash
pytest
```

## Documentation
Install documentation dependencies
```bash
pip install -r requirement-docs.txt
```
In `docs` directory.
```bash
make html
```

## Versioning
Versioneer is used for versioning.
