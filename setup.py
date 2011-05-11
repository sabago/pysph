"""
PySPH
=====

A general purpose Smoothed Particle Hydrodynamics framework.

This package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.  See here:

    http://pysph.googlecode.com

for more information.
"""

from setuptools import find_packages, setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy.distutils.extension import Extension

import numpy
import sys
import os
import multiprocessing
ncpu = multiprocessing.cpu_count()

inc_dirs = [numpy.get_include()]
extra_compile_args = []
extra_link_args = []

mpi_inc_dirs = []
mpi_compile_args = []
mpi_link_args = []

USE_CPP = True
HAS_MPI4PY = True
try:
    import mpi4py
    # assume a working mpi environment
    import commands
    if USE_CPP:
        mpic = 'mpicxx'
    else:
        mpic = 'mpicc'
    mpi_link_args.append(commands.getoutput(mpic + ' --showme:link'))
    mpi_compile_args.append(commands.getoutput(mpic +' --showme:compile'))
    mpi_inc_dirs.append(mpi4py.get_include())
except ImportError:
    HAS_MPI4PY = False

cy_directives = {'embedsignature':True,
                 }

# cython extension modules (subpackage directory:cython file)
extensions = {'base': ['carray.pyx',
                       'point.pyx',
                       'particle_array.pyx',
                       'cell.pyx',
                       'kernels.pyx',
                       'nnps.pyx',
                       'plane.pyx',
                       'polygon_array.pyx',
                       'geometry.pyx',
                       ],
              'sph': ['sph_func.pyx',
                      'sph_calc.pyx',
                      'kernel_correction.pyx',
                      ],
              'sph/funcs': ['basic_funcs.pyx',
                            'position_funcs.pyx',
                            'boundary_funcs.pyx',
                            'external_force.pyx',
                            'density_funcs.pyx',
                            'energy_funcs.pyx',
                            'viscosity_funcs.pyx',
                            'pressure_funcs.pyx',
                            'xsph_funcs.pyx',
                            'eos_funcs.pyx',
                            'adke_funcs.pyx',
                            'arithmetic_funcs.pyx',
                            ],
              'solver': ['particle_generator.pyx',
                         ],
              }

parallel_extensions = {'parallel': ['parallel_controller.pyx',
                                    'parallel_cell.pyx',
                                    'fast_utils.pyx',
                                    ],
                       }

ext_modules = []
for subpkg,files in extensions.iteritems():
    for filename in files:
        path = 'pysph/' + subpkg + '/' + filename
        ext_modules.append(Extension(os.path.splitext(path)[0].replace('/','.'),
                                     ['source/'+path]))

par_modules = []
for subpkg,files in parallel_extensions.iteritems():
    for filename in files:
        path = 'pysph/' + subpkg + '/' + filename
        par_modules.append(Extension(os.path.splitext(path)[0].replace('/','.'),
                                     ['source/'+path]))


if HAS_MPI4PY:
    ext_modules.extend(par_modules)

for extn in ext_modules:
    extn.include_dirs = inc_dirs
    extn.extra_compile_args = extra_compile_args
    extn.extra_link_args = extra_link_args
    extn.pyrex_directives = cy_directives
    if USE_CPP:
        extn.language = 'c++'

for extn in par_modules:
    extn.include_dirs.extend(mpi_inc_dirs)
    extn.extra_compile_args.extend(mpi_compile_args)
    extn.extra_link_args.extend(mpi_link_args)

if 'build_ext' in sys.argv or 'develop' in sys.argv or 'install' in sys.argv:
    d = {'__file__':'source/pysph/base/generator.py'}
    execfile('source/pysph/base/generator.py', d)
    d['main'](None)
    ext_modules = cythonize(ext_modules, nthreads=ncpu, include_path=inc_dirs)

setup(name='PySPH',
      version = '0.9beta',
      author = 'PySPH Developers',
      author_email = 'pysph-dev@googlegroups.com',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      license = "BSD",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",
      packages = find_packages('source'),
      package_dir = {'': 'source'},

      ext_modules = ext_modules,
      
      include_package_data = True,
      cmdclass={'build_ext': build_ext},
      #install_requires=[python>=2.6<3', 'mpi4py>=1.2', 'numpy>=1.0.3', 'Cython>=0.14'],
      #setup_requires=['Cython>=0.14', 'setuptools>=0.6c1'],
      #extras_require={'3D': 'Mayavi>=3.0'},
      zip_safe = False,
      entry_points = """
          [console_scripts]
          pysph_viewer = pysph.tools.mayavi_viewer:main
          """
      )

