Algorithms
==========
.. - Algorithms:
      - Brute Force
      - Barnes-Hut algorithm
      - Quad-Tree
      - Fast multipole method
      - R-Tree
      - R*
      - Convex-Hull

   - Error estimate

N-Body Simulations
------------------
Computing the interactions between agents takes up most of the computational time and resources. This section discusses the state of the problem and some solutions.

.. _Fast Multipole Method: https://www.youtube.com/watch?v=qMLIyZi8Sz0

Brute Force
-----------
Brute force algoritm for iterating distance between all agents is

.. code-block:: python

   for i in range(0, N-1):   # N-1 iterations
      for j in range(i+1, N):  # N-i-1 iterations
         d = distance(x[i], x[j])
         if d < cutoff:
            ...  # Compute force

The number of iterations

.. math::
   \sum_{i=0}^{N-1} (N-i-1) = \frac{(|N| - 1)^2}{2} \in \mathcal{O}(|N|^2)

Since the complexity increases quadratically large simulations become much slower with brute force algorithm. To scale up we need spatial partitioning algorithms.

Spatial Partitioning Algorithms
-------------------------------
Since crowd simulations are only dependent on interactions with agents close by we can partition the space into smaller chunch in order to avoid having to loop with agents far a away.

Algorithm used to compute interactions with near agents is still brute force.

Block List
^^^^^^^^^^
Partitioning into squares


Convex Hull
^^^^^^^^^^^
Partitioning into convex hulls



Parallelisation
---------------
.. note::
   To be done. (Someday)


Refences
--------
.. [partitioning] Vigueras, G., Lozano, M., Orduña, J. M., & Grimaldo, F. (2010). A comparative study of partitioning methods for crowd simulations. Applied Soft Computing Journal, 10(1), 225–235. http://doi.org/10.1016/j.asoc.2009.07.004
