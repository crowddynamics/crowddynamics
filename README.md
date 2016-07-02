# Multi-Agent Simulation of Crowd Dynamics
## Introduction
Continuous 2-dimensional multi-Agent simulation of crowd dynamics. Motion of the agents is modeled using **social force model** where the agents can thought as particles affected by fictitious forces that attempt to produce realistic human movement.


## Python code
Simulation code is written with [_Python_ (3.4)][python] . Code uses [_Numpy_][numpy] for efficient numerical arrays and [_Numba_][numba] to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code readable and easily understandable. 

The aim is to keep the code highly modular and extendable so that new crowd dynamical models can be easily integrated and tested.

Analysing data uses pandas data tables [_Pandas_][pandas].

Graphical user interface for interactive visualization of the simulation uses [_QtPy4_](qtpy4) and [_pyqtgraph_](pyqtgraph).


<!-- Python references -->
[python]: https://www.python.org/
[numpy]: http://www.numpy.org/
[numba]: http://numba.pydata.org/
[pandas]: http://pandas.pydata.org/
[pyqt4]: https://wiki.python.org/moin/PyQt4
[pyqtgraph]: http://www.pyqtgraph.org/ 


## Resources
Timeline of research used for this simulation code.

### 1995
Pioneering research with crowd simulations.

[Social force model for pedestrian dynamics]()
> Helbing, Dirk, and Peter Molnar. "Social force model for pedestrian dynamics." Physical review E 51, no. 5 (1995): 4282.

### 2000
Original social force model.

[Simulating dynamical features of escape panic]()
> Helbing, Dirk, Illés Farkas, and Tamas Vicsek. "Simulating dynamical features of escape panic." Nature 407, no. 6803 (2000): 487-490.

### 2006
Introduction of modeling agent with three circles.

[Crowd dynamics discrete element multi-circle model]()
> Langston, Paul A., Robert Masling, and Basel N. Asmar. "Crowd dynamics discrete element multi-circle model." Safety Science 44, no. 5 (2006): 395-417.

### 2009
Simulation model details.

[Fire Dynamics Simulator with Evacuation: FDS+Evac]()
> Korhonen, Timo, and Simo Hostikka. "Fire dynamics simulator with evacuation: FDS+ Evac." Technical Reference and User’s Guide. VTT Technical Research Centre of Finland (2009).


[Modelling subgroup behaviour in crowd dynamics DEM simulation]() 
> Singh, Harmeet, Robyn Arter, Louise Dodd, Paul Langston, Edward Lester, and John Drury. "Modelling subgroup behaviour in crowd dynamics DEM simulation." Applied Mathematical Modelling 33, no. 12 (2009): 4408-4423.

### 2010

Model for counterflow situtations in crowd simulations.

[Counterflow model for agent-based simulation of crowd dynamics]() 
> Korhonen, T., S. Heliovaara, Simo Hostikkaa, and H. Ehtamo. "Counterflow model for agent-based simulation of crowd dynamics." Safety Science (2010).

### 2013
Game theoretical approach for human behaviour in egress congestion.

[Patient and impatient pedestrians in a spatial game for egress congestion]()
> Heliövaara, Simo, Harri Ehtamo, Dirk Helbing, and Timo Korhonen. "Patient and impatient pedestrians in a spatial game for egress congestion." Physical Review E 87, no. 1 (2013): 012802.

### 2014
Improved algorithm for social force based on human anticipatory behavior.

[Universal Power Law Governing Pedestrian Interactions](http://motion.cs.umn.edu/PowerLaw/)
> Karamouzas, Ioannis, Brian Skinner, and Stephen J. Guy. "Universal power law governing pedestrian interactions." Physical review letters 113, no. 23 (2014): 238701.

### Simulation models and Software

[Fire Dynamics Simulator and Smokeview](https://pages.nist.gov/fds-smv/)

[PedSim](http://pedsim.silmaril.org/)

[Simulex](https://www.iesve.com/software/ve-for-engineers/module/Simulex/480) 

[Golaem Crowd](http://golaem.com/crowd) 

[MASSIVE (software)](http://www.massivesoftware.com/) 

[Legion](http://www.legion.com/) 

### Web resources

[UNC gamma](http://gamma.cs.unc.edu/research/crowds/) 

[Crowd Simulation Group](http://www.crowdsimulationgroup.co.uk/)
