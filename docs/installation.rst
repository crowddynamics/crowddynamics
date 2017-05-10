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

.. |Appveoyr| image:: https://ci.appveyor.com/api/projects/status/nlqrc850nbr9kh4e?svg=true
   :target: https://ci.appveyor.com/project/jaantollander/crowddynamics
   :alt: Appveoyr continuous intergration


Conda
-----
.. note::

   Conda package has not yet been released. It will be released as soon as crowddynamics releases version 0.0.1.


Source
------

.. raw:: html

    <div style="position:relative;height:0;padding-bottom:56.25%"><iframe src="https://www.youtube.com/embed/IN63QLZBN2U?ecver=2" style="position:absolute;width:100%;height:100%;left:0" width="640" height="360" frameborder="0" allowfullscreen></iframe></div>


.. note::

   Run all of the commands in the project root directory ``<path>/crowddynamics`` after you have downloaded the source code.


1) Download the source code with ``git``

  .. code-block:: bash

     git clone https://github.com/jaantollander/CrowdDynamics.git

2) Download the `Anaconda <https://www.continuum.io/downloads>`_ package manager or `Miniconda <http://conda.pydata.org/miniconda.html>`_ which is the same as Anaconda except it does not install 150 scientific packages with the installation.

  After the installation use ``conda`` to install a new environment and the dependensies to run ``crowddynamics``

  .. code-block:: bash

     conda env create python3.5 -n crowd35 -f environment.yml
     source activate crowd35

3) Then install ``crowddynamics`` itself in ``editable`` mode using following ``pip`` command

  .. code-block:: bash

     pip install --editable .

4) Now you can run ``crowddynamics`` from the commandline. Check out the available commands using

  .. code-block:: bash

     crowddynamics --help
