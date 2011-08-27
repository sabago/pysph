.. _linux_install:

--------------------------
Linux Installation
--------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Supported Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PySPH is tested on the following operating system versions:

+     Ubuntu (9.10, 10.04, 10.10) 32 and 64 bit
+     Fedora (14) 64 bit


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following packages need to be installed on your system. Contact
your system administrator if you do not have administrative rights.

+ Setuptools_ to install python dependencies via `easy_install`
+ Numpy_ : >= 1.3
+ Virtualenv_ to create your isolated Python environment. 
+ MPI_ to enable PySPH to run in parallel

Additionally, you need Mercurial_ installed if you want to check out
the latest development version of PySPH.

For Ubuntu systems, you can install these like so::

    sudo apt-get install python-setuptools python-virtualenv mercurial openmpi libopenmpi-dev python-numpy

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Setting up your environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend using a virtual environment for your PySPH installation so
that all PySPH specific dependencies do not interfere with other
python packages.

Your virtualenv may be set up like so::

     mkdir ~/envs
     virtualenv ~/envs/pysph
     source ~/envs/pysph/bin/activate

You can now install PySPH for your virtual environment::

     easy_install mayavi2 pysph

If you want to build PySPH from source, you will need Cython_. We
recommend you also install Sphinx_ to build the documentation and
Nose_ to run the tests. PySPH comes with a default application to view
simulation results interactively. This requires Mayavi_ to be
installed on your system.

These packages can be installed like so::

       easy_install cython nose sphinx mpi4py mayavi2

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing PySPH from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming all the previous steps completed without any error, you can
download PySPH and install it like so::

	 hg clone https://pysph.googlecode.com/hg ~/pysph
	 cd ~/pysph
	 python setup.py install

	 
.. note::
   
   If you are on a Fedora machine and the system administrator has
   installed the openmpi_ and the mpich2_ implementations for MPI_,
   you may need to unload the mpich2 module and load the openmpi
   module.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Running the tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

   Ignore this step if you installed PySPH with setuptools_ and
   easy install.

We recommend you test the fitness of your PySPH installation by
running the test suite::

	make testall

All tests should pass apart from a few which are marked as skipped.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Running the examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can now run the examples like so::

    cd examples/stress
    python stress2d_balls.py &
    pysph_viewer

If you have mpi4py_ installed, you can run the same example in
parallel like so::

    mpirun -n 2 ~/envs/pysph/bin/python stress2d_balls.py

The output for the example will be in the directory
`stress2d_balls_output` by default.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Building the docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PySPH uses the Sphinx_ documentation system. The docs can be built
like so::

     cd ~/pysph/docs
     make html

.. _mpi4py: http://mpi4py.scipy.org

.. _openmpi: http://www.open-mpi.org

.. _mpich2: http://www.mcs.anl.gov/research/projects/mpich2

.. _MPI: http://www.mcs.anl.gov/research/projects/mpi

.. _Setuptools: http://pypi.python.org/pypi/setuptools

.. _Numpy: http://numpy.scipy.org

.. _Virtualenv: http://pypi.python.org/pypi/virtualenv

.. _Mercurial: http://mercurial.selenic.com

.. _Sphinx: http://sphinx.pocoo.org/

.. _Mayavi: http://code.enthought.com/projects/mayavi

.. _Cython: http://cython.org

.. _Sphinx: http://sphinx.pocoo.org/

.. _Nose: http://www.somethingaboutorange.com/mrl/projects/nose

..  LocalWords:  mpi openmpi Setuptools Virtualenv
