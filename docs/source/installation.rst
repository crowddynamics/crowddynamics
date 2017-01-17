Installation
============

Developers
----------

.. note::

   Run all of the commands in the project root directory ``<path>/crowddynamics`` after you have downloaded the source code.


1) Download the source code with ``git``

  .. code-block:: bash

     git clone https://github.com/jaantollander/CrowdDynamics.git

2) Download the `Anaconda <https://www.continuum.io/downloads>`_ package manager or `Miniconda <http://conda.pydata.org/miniconda.html>`_ which is the same as Anaconda except it does not install 150 scientific packages with the installation.

  After the installation use ``conda`` to install a new environment and the dependensies to run ``crowddynamics``

  .. code-block:: bash

     conda env create python3.4 -n crowd34 -f environment.yml
     source activate crowd34

3) Then install ``crowddynamics`` itself in ``editable`` mode using following ``pip`` command

  .. code-block:: bash

     pip install --editable .

4) Now you can run ``crowddynamics`` from the commandline. Check out the available commands using

  .. code-block:: bash

     crowddynamics --help
