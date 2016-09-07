"""
Profiling and Benchmarks
========================

kernprof -l test.py

python -m line_profiler test.py.lprof


Jit compiling speed


.. [#] http://www.scipy-lectures.org/advanced/optimizing/
.. [#] http://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/

"""
import sys
sys.path.insert(0, "/home/jaan/Dropbox/Projects/CrowdDynamics")

from src.main import run_simulation

run_simulation("room_evacuation")
