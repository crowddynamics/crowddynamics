Multi-Agent Simulation
======================
## Introduction

Multi-Agent simulation of crowd dynamics. Movement is modeled with social force model by _Helbing_ [1] combined with improved model _power law for pedestrian interactions_ [2].

Behaviour in evacuation situations is modeled with _game theoretical model_ [3] for egress congestion. 


## Python code
Simulation code is written with _Python 3.4_. Code uses _Numpy_ for efficient numerical arrays and _Numba_ to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code very readable. The aim is to keep the code highly modular so that new crowd dynamical models can be easily integrated and tested.


## Resources
### 2000
- [Simulating dynamical features of escape panic](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.323.245&rep=rep1&type=pdf)

### 2009
- [Fire Dynamics Simulator with Evacuation: FDS+Evac](http://www.vtt.fi/inf/pdf/workingpapers/2009/W119.pdf)

### 2011
- [Patient and impatient pedestrians in a spatial game for egress congestion](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.87.012802)

### 2014
- [Universal Power Law Governing Pedestrian Interactions](http://motion.cs.umn.edu/PowerLaw/)

<!--- [Experimental evidence of the "Faster Is Slower" effect](http://www.sciencedirect.com/science/article/pii/S2352146514001215)-->

<!--- [Optimal Reciprocal Collision Avoidance (ORCA)](http://gamma.cs.unc.edu/ORCA/)-->

## Some existing simulation models

- [Fire Dynamics Simulator and Smokeview](https://github.com/firemodels/fds-smv)

