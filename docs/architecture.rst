Architecture
============

API
---
Crowddynamics API is implemented using traitlets_ and traitypes_. Traitlets is an framework that lets classes have attributes with

* Type checking and
* Validation
* Dynamically calculated default values (along with normal static ones)
* *"On change"* callback functions
* Configuration mechanism loading values either from

    - Python file
    - JSON format
    - Commandline arguments

These attributes are referred as traits. The information about the attribute given by the trait can be used for

* Generating documentation
* Generating commandline arguments
* Generating GUI widgets
* Controlling which attributes are saved into disk


Numerical Core
--------------

- numpy
- numba
- scipy
- scikits
- shapely



.. _traitlets: https://traitlets.readthedocs.io/en/stable/
.. _traitypes: https://traittypes.readthedocs.io/en/latest/?badge=latest
