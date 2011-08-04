.. _tests:

-----------------------
Running the tests
-----------------------

To verify your PySPH installation, it recommended to run the
tests. The number of tests actually run will depend on your
installation. For example, if you do not have PyOpenCL installed, all
tests that require this library will be *Skipped*. PySPH uses
nosetests_ for testing and it is assumed that it is installed on your
system. Follow the link to install it if it is not.


.. note::
   Long running tests are labelled *slow* in PySPH and may be omitted in
   routine testing. It is recommended however, to run **all** these 
   tests at least once. For example,  when you obtain a new release.

^^^^^^^^^^^^^^^^^^^^^^^^^^
Liunx
^^^^^^^^^^^^^^^^^^^^^^^^^^

The tests can be run like so::

    $ make test

This runs all tests in PySPH that are not labelled as *slow*. To run
all tests regardless of their label::

    $ make testall

You should see some very pleasing dots and a small statistic for the
number of tests run.

.. warning::
   If an error is reported or worse yet, a test fails, it means something
   is not right. You should report it to the 
   PySPH developers at pysph-dev@googlegroups.com
	
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a Windows machine, you have to use the :command:`nosetests` command
like so::

     $ nosetests -v -a '!slow' source

to run all tests that are not *slow*, and::

   $ nosetests -v source

to run **all** tests.

You should see some nice looking dots and a small statistic for the
number of tests run.

.. warning::
   If an error is reported or worse yet, a test fails, it means something
   is not right. You should report it to the 
   PySPH developers at pysph-dev@googlegroups.com


.. _nosetests: http://www.somethingaboutorange.com/mrl/projects/nose
