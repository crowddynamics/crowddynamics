Getting Started
===============

Installing dependensies
-----------------------
.. note::
   Crowd Dynamics still has to be run from the source. Setup file is still unfinished.

Virtualenv
^^^^^^^^^^
.. warning::
   Installation with virtualenv is stil untested.

Install virtualenv to your Python installation using

::

   pip install virtualenv

Make new environment using and activate it

::

   virtualenv crowd35
   source crowd35/bin/activate

Then install dependensies in the project root directory with commands

::

   pip install -r requirements.txt

To deactivate virtualenv

::

   source deactivate

Anaconda
^^^^^^^^
With `Anaconda`_ package manager in run commands

.. _Anaconda: https://www.continuum.io/downloads


::

   conda env create -f buildscripts/environment.yaml
   source activate crowd35



Examples
--------

.. note::
   To be written
