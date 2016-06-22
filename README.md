# Multi-Agent Simulation
## Introduction

Multi-Agent simulation of crowd dynamics. Movement is modeled with social force model by _Helbing_ combined with improved model _power law for pedestrian interactions_.

Behaviour in evacuation situations is modeled with _game theoretical model_ for egress congestion. 


## Python code
Simulation code is written with [_Python_][python] using version _3.4_. Code uses [_Numpy_][numpy] for efficient numerical arrays and [_Numba_][numba] to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code readable and easily understandable. 

The aim is to keep the code highly modular and extendable so that new crowd dynamical models can be easily integrated and tested.

[python]: https://www.python.org/
[numpy]: http://www.numpy.org/
[numba]: http://numba.pydata.org/


## Resources
Timeline of research used for this simulation code.

### 2000
Pioneering research with crowd simulations.
[Simulating dynamical features of escape panic](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.323.245&rep=rep1&type=pdf)

### 2005
Modeling agent with three circles.
[Crowd dynamics discrete element multi-circle model](http://www.sciencedirect.com/science/article/pii/S0925753505001724)

### 2009
Simulation model details.
[Fire Dynamics Simulator with Evacuation: FDS+Evac](http://www.vtt.fi/inf/pdf/workingpapers/2009/W119.pdf)

[Modelling subgroup behaviour in crowd dynamics DEM simulation](http://www.sciencedirect.com/science/article/pii/S0307904X09000808) 

### 2011

Model for counterflow situtations in crowd simulations.
[Counterflow model for agent-based simulation of crowd dynamics](http://www.sciencedirect.com/science/article/pii/S0360132311002630) 

### 2013
Game theoretical approach for human behaviour in egress congestion.
[Patient and impatient pedestrians in a spatial game for egress congestion](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.87.012802)

### 2014
Improved algorithm for social force based on human anticipatory behavior.
[Universal Power Law Governing Pedestrian Interactions](http://motion.cs.umn.edu/PowerLaw/)

### Simulation models
[Fire Dynamics Simulator and Smokeview](https://github.com/firemodels/fds-smv)

[PedSim](http://pedsim.silmaril.org/)

[Simulex](https://www.iesve.com/software/ve-for-engineers/module/Simulex/480) 

