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

   * - Product
     -
     -
   * - Anaconda_
     - Scientific Python distribution
     -

Anaconda is used for managing Python environments. Anaconda can install precompiled binaries which is very useful for installing dependencies required by ``crowddynamics``. It makes installation process faster and more reliable because users don't have to compile external C dependencies themselves. This also relieves the effort of making crowddynamics available on Windows, which often different C compiler that Linux or might lack one.


Editors
-------
.. list-table:: Software Tools
   :header-rows: 1

   * - Product
     - Type
     -
   * - PyCharm_
     - Desktop GUI application
     -
   * - Atom_
     - Desktop GUI application
     -
   * - Jupyter_
     - Browser Application
     -
   * - Kite_
     - Intelligent code assistant
     -


Choice of editor comes to mostly to personal preference. I personally recommend PyCharm for which students can get the pro version for free. PyCharm is excellent editor that gives intelligent code suggestions and integrates other useful tools such as debugger and version control into one tools.

Also using terminal based editor such as VIM or emacs should work fine.


Version Control
---------------
.. list-table:: Software Management Tools and Services
   :header-rows: 1

   * - Tool
     -
     -
   * - Git_
     - Version Control System
     -
   * - GitHub_
     - Hosting Repository
     -
   * - TravisCI_
     - Continuous integration for linux
     -
   * - Appveyor_
     - Continuous integration for windows
     -


Documenting
-----------
.. list-table:: Documenting Tools
   :header-rows: 1

   * - Tool
     -
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
   * - graphviz_
     - Graph visualization tool
   * - tikz_
     - Tool for producing vector graphics using tex macros.
   * - bokeh_
     - Python interactive visualization library.


Documenting the source code


.. list-table:: Video Documentation Tools
   :header-rows: 1

   * - Tool
     -
   * - SimpleScreenRecorder_
     - Simple free video recording tool for Linux.
   * - Kdenlive_
     - Video editing tool for Linux.


Video documentation can be published to Youtube.


Testing
-------
.. list-table::
   :header-rows: 1

   * -
     -
   * - pytest_
     - Python testing framework
   * - pytest-cov_
     - Pytest plugin for measuring coverage
   * - pytest-benchmark_
     - Pytest plugin for benchmarking code
   * - hypothesis_
     - Pytest plugin for property based testing


Task Management
---------------
.. list-table::
   :header-rows: 1

   * -
     -
   * - doit_
     - Task automation tool


.. Links
.. _Ubuntu: https://www.ubuntu.com/
.. _Anaconda: https://www.continuum.io/
.. _PyCharm: https://www.jetbrains.com/pycharm/
.. _Atom: https://atom.io/
.. _Jupyter: https://jupyter.org/
.. _Kite: https://kite.com/
.. _Git: https://git-scm.com/
.. _GitHub: https://github.com/
.. _TravisCI: https://travis-ci.org/
.. _Appveyor: https://www.appveyor.com/
.. _ReStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://www.sphinx-doc.org/en/stable/
.. _sphinx_rtd_theme: https://github.com/rtfd/sphinx_rtd_theme
.. _LaTeX: https://www.latex-project.org/
.. _GitHubPages: https://pages.github.com/
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