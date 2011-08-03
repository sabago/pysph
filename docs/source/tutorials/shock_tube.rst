.. _shock_tube:

===============================
The shock tube problem
===============================
    
The shock tube problem was one of the first extensions of SPH outside
the field of astrophysics, for which it was developed
[Monaghan1983]_. The equations solved are the inviscid Navier Stokes
equations or the Euler equations. The shock tube problem is the
starting point for schemes that attempt to resolve discontinuities
like shocks and entropy waves.

.. _euler-equations:

----------------------------------
Problem definition
----------------------------------

The shock tube problem consists of a tube of fluid that is initially
at rest. A central diaphragm in the tube separates two states of the
fluid. The fluid to the left has a higher pressure and energy as
compared with the fluid on the right. We study the evolution of the
fluid as the diaphragm is ruptured instantaneously.

The exact solution to this problem is known.. It consists of a shock
wave moving to the right, a contact discontinuity moving with the
speed of the fluid to the right and a rarefaction moving to the left
as shown in the figure.

.. _shock_exact:
.. figure:: images/shock-exact.png
    :align: center
    :width: 350


The continuous form of the Euler equations are:

the continuity equation:

.. math::

    \frac{D\rho}{Dt} = -\rho\nabla \, \cdot \vec{v},

the momentum equation:

.. math::
   \frac{D\vec{v}}{Dt} = -\frac{1}{\rho}\nabla(P),

the thermal energy equation:

.. math::
   \frac{De}{Dt} = -\left( \frac{P}{\rho} \right)\,\nabla\,\cdot \vec{v},

and the ideal gas equation of state:

.. math::

   p = (\gamma -1)\rho e   

----------------------------------------
SPH Equations
----------------------------------------

The SPH discretization for the Euler equations (:ref:`euler-equations`) are:

.. math::

   p = \rho(\gamma - 1)e

   \rho_a = \sum_{b=1}^{N} m_b\,W_{ab}

   \frac{Du_a}{Dt} = -\sum_{b=1}^{N}m_b\left( \frac{P_a}{\rho_a^2} + \frac{P_b}{\rho_b^2} + \Pi_{ab} \right )\,\nabla_a\,W_{ab}

   \frac{De_a}{Dt} = \frac{1}{2}\sum_{b=1}^{N}m_b\left( \frac{P_a}{\rho_a^2} + \frac{P_b}{\rho_b^2} + \Pi_{ab} \right )\cdot\,\nabla_aW_{ab}

   \frac{D{x_a}}{Dt} = u_a

To handle shocks that may develop in the solution, an artificial
viscosity term, :math:`\Pi_{ab}` is added to the momentum and thermal
energy equation [Monaghan1992]_

---------------------------------
Boundary conditions
---------------------------------

The dam break problem is simulated without any boundary
conditions. Typically, the domain of interest is chosen large enough
so that the errors from the boundary do not propagate into the region
of interest.

----------------------------------
Running the example 
----------------------------------

The example code for this problem is located in the
:file:`examples/shock-tube` directory and can be run like so::

	$ cd examples/shock-tube
	$ python shock_tube.py

This will create a solution directory :file:`shock_tube_output` and by
default, dump output files every twenty iterations. Each output file
name is of the form :file:`shock_tube_0_count.npz` where, *count* is
the iteration count for that particular file.

--------------------------------------
Results
--------------------------------------

Results for the shock tube problem are usually depicted as a plot of a
primitive variable (:math:`u, \,\, \rho, \,\, p`) versus
distance. 

Recall that the output files were put in a directory
:file:`shock_tube_output`. Move to that directory and launch the
Python interpreter::

    $ cd shock_tube_output
    $ ipython -pylab

Execute the following in the interpreter:

.. sourcecode:: python

   import pysph.solver.api as solver

   data = solver.load("shock_tube_0_500.npz")
   array = data["arrays"]["fluid"]
   solver_data = data["solver_data"]
   plot(array.p, array.x)
   xlim(-.4,.4)

   dt = solver-data["dt"]
   title(r"Pressure at $t = %f$"%(dt))

to produce the :ref:`shock-tube-sample-plot`

.. _shock-tube-sample-plot:
.. figure:: images/shock-tube-pressure-plot.png
    :align: center
    :width: 500

    Example plot
