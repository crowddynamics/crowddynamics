Installation
============

.. list-table:: Supported Platforms
   :header-rows: 1

   * - Platform
     - Support
   * - Linux
     - |Travis|
   * - Windows
     - |Appveoyr|
   * - OSX
     - Not tested

.. |Travis| image:: https://travis-ci.org/jaantollander/crowddynamics.svg?branch=master
   :target: https://travis-ci.org/jaantollander/crowddynamics
   :alt: Travis continuous intergration

.. |Appveoyr| image:: https://ci.appveyor.com/api/projects/status/2d9nsf41xjcpn0ka?svg=true
   :target: https://ci.appveyor.com/project/jaantollander/crowddynamics-wi50b
   :alt: Appveoyr continuous intergration


Using Conda
-----------
.. note::

   CrowdDynamics has not yet been released as a Conda package. It will be released as soon as crowddynamics releases version 0.0.1.


From Source
-----------

.. raw:: html

    <div style="position:relative;height:0;padding-bottom:56.25%"><iframe src="https://www.youtube.com/embed/IN63QLZBN2U?ecver=2" style="position:absolute;width:100%;height:100%;left:0" width="640" height="360" frameborder="0" allowfullscreen></iframe></div>


Install Conda Package Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install *Conda* package manager. `Miniconda <http://conda.pydata.org/miniconda.html>`_ is faster to install than the full `Anaconda <https://www.continuum.io/downloads>`_ distribution which comes bundled with lots of other useful scientific packages, but are not required to run ``crowddynamics``.


Download the Source Code
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Run all of the commands in the project root directory ``crowddynamics`` after you have downloaded the source code.


1) Download the source code with ``git``

  .. code-block:: bash

     git clone https://github.com/jaantollander/crowddynamics.git

2) After the installation use ``conda`` to install a new environment and the dependensies to run ``crowddynamics``

  .. code-block:: bash

     conda env create python3.5 -n crowd35 -f environment.yml
     source activate crowd35

3) Then install ``crowddynamics`` itself in ``editable`` mode using following ``pip`` command

  .. code-block:: bash

     pip install --editable .

4) Now you can run ``crowddynamics`` from the commandline. Check out the available commands using

  .. code-block:: bash

     crowddynamics --help


Run Tests
^^^^^^^^^
Install dependencies to run the tests::

    pip install -r requirements-tests.txt

Run test suite using ``pytest``::

     py.test
