# Crowd Dynamics
![](docs/figures/crowddynamics-gui.png)

Crowd dynamics is a simulation environment written in Python package for simulation movement of crowds. The project was created in summer 2016 for Systems Analysis Laboratory at Aalto University in Finland. The [documentation](http://jaantollander.github.io/crowddynamics/) has more detail about the project.


## Installation
Crowd dynamics is tested on Ubuntu 16.04 using Python 3.6. First, install the Conda package manager. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution is the easiest to install. Then, clone the `crowddynamics` repository.
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
We can use a graphical user interface (GUI) for visualizing the crowd simulations and display data interactively, which can be very useful for designing new simulations and debugging. We have implemented the [GUI for crowddynamics](https://github.com/jaantollander/crowddynamics-qtgui) using Qt via `pyqt` and `pyqtgraph`. It is maintained separately from the `crowddynamics` repository due to its dependencies on Qt.

We start by cloning the repository. 
```bash
git clone https://github.com/jaantollander/crowddynamics-qtgui.git
```

Inside `crowddynamics-qtgui` directory, install the requirements and install `crowddynamics-qtgui` as editable. We must install `pyqt` version 4 using Conda from the `conda-force` channel for it to work correctly. Be sure that you activate the environment where you installed `crowddynamics`.  
```bash
conda activate crowd36
conda install pyqt=4 -c conda-forge
pip install -r requirements.txt
pip install --editable .
```

## Usage
Inside the `crowddynamics/examples` directory, we can find example simulations.

```python
from crowddynamics.examples.simulations import Hallway
from crowddynamics.simulation.agents import Circular
from crowddynamics.logging import setup_logging

if __name__ == '__main__':
    setup_logging()
    iterations = 1000
    simulation = Hallway(agent_type=Circular)
    for i in range(iterations):
        simulation.update()
```

## Tests
Tests are implemented using [Pytest](https://docs.pytest.org/en/latest/) and [Hypothesis](https://hypothesis.readthedocs.io/en/latest/).

Install test dependencies.
```bash
pip install -r requirements-tests.txt
```
In the project's root directory, run `pystest` test suite.
```bash
pytest
```

## Documentation
Documentation is created using [Sphinx](https://www.sphinx-doc.org/en/master/).

Install documentation dependencies
```bash
pip install -r requirement-docs.txt
```
In `docs` directory.
```bash
make html
```

## Versioning
[Versioneer](https://github.com/warner/python-versioneer) is used for versioning.
