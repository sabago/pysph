.. _quick_install:

---------------------------
Quick Installation Guide
---------------------------

Read this guide if you want to get up and running with PySPH in under
five minutes. 
Refer to the complete installation :doc:`guide <complete_install_guide>`
if you need detailed instructions. 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting the code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code for PySPH is publicly available here_ and the repository may
 be checked out like so::

	hg pull https://pysph.googlecode.com/hg pysph

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most basic dependency for PySPH is Cython_ which is required to
compile the code. In addition to this, we recommend you install Sphinx_
to build the documentation and Nose_ to run the tests. These three
packages can be installed like so::

	 easy_install cython sphinx nose

.. _Sphinx: http://sphinx.pocoo.org/

.. _Nose: http://www.somethingaboutorange.com/mrl/projects/nose

.. _Cython: http://cython.org

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing PySPH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the dependencies are met, installation is as simple as::

     cd pysph
     python setup.py install

You can now run the examples like so::
      
      cd examples/dam_break
      python dam_break.py

This will run a 2D dam break problem and store the output in the
directory `dam_break_output`.


.. _here: http://www.code.google.com/p/pysph
