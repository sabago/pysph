# Installation #

The current installation instructions are here:

https://bitbucket.org/pysph/pysph/overview

and here

http://pysph.readthedocs.org/en/latest/installation.html


What follows are old instructions for the older 0.9 release.


---


## Old instructions ##


There are several ways to install PySPH. You need to make sure you have the following dependencies installed.

## Dependencies ##

The following dependencies are absolutely essential:

  * numpy: version 1.0.4 and above will work.
  * Cython: version 0.12 and above. This implies that your system have a working C compiler.
  * mpi4py: version 1.2 and above. This will require working MPI libs on your computer
  * setuptools: version 0.6c9 is tested to work. This is needed to build and install the package. A later version will work better.

The following dependencies are optional but recommended:

  * nose: This package is used for running any tests.
  * Sphinx: version 0.5 will work. This package is used for building the documentation.
  * VTK: versions, 4.x, 5.x and above will work. This is used to write VTK files for subsequent output and visualization.
  * Mayavi: version 3.x and above. This is convenient for visualization and generation of VTK files. This is entirely optional though.

Dependencies on Debian/Ubuntu

Installing the dependencies on a recent flavor of Ubuntu or Debian is relatively easy. Just do the following:

```
$ sudo apt-get install python-numpy python-setuptools python-dev gcc python-nose mayavi2 libopenmpi-dev
```

Cython and Sphinx may not be readily available, in which case you can install them using either the usual python setup.py dance for the respective projects or using the more convenient:

```
$ easy_install Cython
$ easy_install mpi4py
```

Dependencies on Fedora:

```
# yum install numpy python-setuptools python-devel gcc Cython python-nose mpi4py-openmpi openmpi-devel
```

Once you have the essential dependencies installed you can easily build the package.


## Install from Sources ##

Currently a tarball is not released, so you need to do a hg checkout/clone of the project:

```
$ hg clone https://pysph.googlecode.com/hg/ pysph 
```

and then cd into the pysph directory and then issue:

```
$ python setup.py install
```

If you’d like a developer install, which is preferred in case you need to make any changes and develop on PySPH, you can develop as you go do the following:

```
$ python setup.py develop
```

Once this is done, you can use the package and then when you change a file in your branch you can either recompile inplace or re-issue the develop command as:

```
$ python setup.py build_ext --inplace
```

It is recommended to install PySPH in virtualenv so that your entire PySPH install is isolated inside the environment created by virtualenv and uninstalling is as easy as deleting the directory.
Assuming you have Python installed along with virtualenv, numpy, and a working MPI setup you can do the following:

```
$ virtualenv pysph
$ source pysph/bin/activate
$ easy_install Cython
$ easy_install mpi4py
$ cd PySPH-source-directory
$ python setup.py install
```

You may also install the optional dependencies like Mayavi, like so:

```
$ easy_install Mayavi[app]
```

Running the tests

Once the package is installed you can test if everything is OK by running the test suite like so:

```
$ cd source
$ nosetests --exe
```

If this runs without error you are all set. If not please contact the developers.

PS: There is devel clone at http://code.google.com/p/pysph/source/browse?repo=devel which is the one undergoing development currently and will be merged into the main branch it is deemed sufficiently complete.