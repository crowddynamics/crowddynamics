Developing
==========
.. note::
   We assume that developers have the basic knowledge of commandline.


Operating System
----------------
Using *Linux* is recommended. Latest Ubuntu_ distribution should work well. Windows still remains untested and might not fully work.


Programming
-----------
.. list-table::
   :header-rows: 1

   * - Tool
     - Description
   * - Anaconda_
     - Scientific Python distribution

Anaconda is used for managing Python environments. Anaconda can install precompiled binaries which is very useful for installing dependencies required by ``crowddynamics``. It makes installation process faster and more reliable because users don't have to compile external C dependencies themselves. This also relieves the effort of making crowddynamics available on Windows, which often different C compiler that Linux or might lack one.


Editors
-------
.. list-table:: Software Tools
   :header-rows: 1

   * - Tool
     - Description
   * - PyCharm_
     - IDE for Python development from JetBrains.
   * - Atom_
     - Open-source text and code editor for GitHub.
   * - Jupyter_
     - Open-source notebook that can contain live code, equations, text and visualizations.
   * - Kite_
     - Intelligent code assistant that leverages code on the web.


Choice of editor comes to mostly to personal preference. I personally recommend PyCharm for which students can get the pro version for free. PyCharm is excellent editor that gives intelligent code suggestions and integrates other useful tools such as debugger and version control into one tools.

Also using terminal based editor such as VIM or emacs should work fine.


Version Control
---------------
.. list-table:: Software Management Tools and Services
   :header-rows: 1

   * - Tool
     - Description
   * - Git_
     - Version Control System
   * - GitHub_
     - Hosting Repository
   * - Versioneer_
     - Version string management
   * - TravisCI_
     - Continuous integration for linux
   * - Appveyor_
     - Continuous integration for windows


Crowddynamics_ repository is hosted at GitHub. Automated testing is handle by continuous integration services that are integrated.



Documenting
-----------
.. list-table:: Documenting Tools
   :header-rows: 1

   * - Tool
     - Description
   * - ReStructuredText_
     - Markup syntax used for writing documentation
   * - LaTeX_
     - High-quality typesetting system used for writing scientific documentation.
   * - Sphinx_
     - Python documentation tool.
   * - sphinx_rtd_theme_
     - Theme used for HTML documentation.
   * - `Google Style Docstring <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
     - Style for documenting docstring of functions and classes.
   * - GitHubPages_
     - Static website hosted from the projects GitHub repository.


The documentation of the project is located under ``docs`` directory. Documentation is compiled using ``make html`` command inside ``docs`` directory. Sphinx has autodoc_ extension that can pull documentation from docstring inside crowddynamics source code. This feature is extensively used to document the project.



.. list-table:: Visualization Tools
   :header-rows: 1

   * - Tool
     - Description
   * - GeoGebra_
     - Tool for drawing and manipulating geometric object.
   * - graphviz_
     - Graph visualization tool
   * - tikz_
     - Tool for producing vector graphics using tex macros.
   * - bokeh_
     - Python interactive visualization library.


.. list-table:: Video Documentation Tools
   :header-rows: 1

   * - Tool
     - Description
   * - SimpleScreenRecorder_
     - Simple free video recording tool for Linux.
   * - Kdenlive_
     - Video editing tool for Linux.


Also documenting how to type videos and demonstrations into video is import. Above list contains some open source tools for Linux to record and edit videos. Videos can be published for example in Youtube.


Testing
-------
.. list-table:: Testing Tools
   :header-rows: 1

   * - Tool
     - Description
   * - pytest_
     - Python testing framework
   * - pytest-cov_
     - Pytest plugin for measuring coverage
   * - pytest-benchmark_
     - Pytest plugin for benchmarking code
   * - hypothesis_
     - Pytest plugin for property based testing

Testing is important part of any software development process. Crowddynamics uses pytest framework for testing the source code. It also uses several plugins to extend the capabilities to measure code coverage, benchmarking and property based testing. Tests for each module are written inside ``crowddynamics`` ``tests`` module so that every module contains its own tests.


Task Management
---------------
.. list-table::
   :header-rows: 1

   * - Tool
     -
   * - doit_
     - Task automation tool


Distributing
------------
.. todo:: Distributing crowddynamics through conda


Research
--------
.. list-table::
   :header-rows: 1

   * - Tool
     -
   * - Mendeley_
     -
   * - Zotero_
     -
   * - ResearchGate_
     -
   * - Arxiv_
     -


.. _Mendeley: https://www.mendeley.com/
.. _Zotero: https://www.zotero.org/
.. _ResearchGate: https://www.researchgate.net
.. _Arxiv: https://arxiv.org/


.. Links
.. _crowddynamics: https://github.com/jaantollander/crowddynamics
.. _Ubuntu: https://www.ubuntu.com/
.. _Anaconda: https://www.continuum.io/
.. _PyCharm: https://www.jetbrains.com/pycharm/
.. _Atom: https://atom.io/
.. _Jupyter: https://jupyter.org/
.. _Kite: https://kite.com/
.. _Git: https://git-scm.com/
.. _GitHub: https://github.com/
.. _Versioneer: https://github.com/warner/python-versioneer
.. _TravisCI: https://travis-ci.org/
.. _Appveyor: https://www.appveyor.com/
.. _ReStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://www.sphinx-doc.org/en/stable/
.. _sphinx_rtd_theme: https://github.com/rtfd/sphinx_rtd_theme
.. _LaTeX: https://www.latex-project.org/
.. _GitHubPages: https://pages.github.com/
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _SimpleScreenRecorder: http://www.maartenbaert.be/simplescreenrecorder/
.. _Kdenlive: https://kdenlive.org/
.. _graphviz: http://www.graphviz.org/
.. _tikz: https://en.wikipedia.org/wiki/PGF/TikZ
.. _bokeh: http://bokeh.pydata.org/en/latest/
.. _pytest: https://docs.pytest.org/en/latest/
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _pytest-benchmark: https://readthedocs.org/projects/pytest-benchmark/
.. _hypothesis: http://hypothesis.works/
.. _doit: http://pydoit.org/
.. _GeoGebra: https://www.geogebra.org/