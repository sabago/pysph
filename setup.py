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
HAS_CYTHON=True
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    cmdclass = {'build_ext': build_ext}
except ImportError:
    HAS_CYTHON=False
    cmdclass = {}

from numpy.distutils.extension import Extension

import numpy
import sys
import os
import platform
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

C_EXTN = 'c'
if USE_CPP:
    C_EXTN = 'cpp'

# cython extension modules (subpackage directory:cython file)
extensions = {'base': ['carray.pyx',
                       'fast_utils.pyx',
                       'point.pyx',
                       'particle_array.pyx',
                       'cell.pyx',
                       'kernels.pyx',
                       'nnps.pyx',
                       'plane.pyx',
                       'polygon_array.pyx',
                       'geometry.pyx',
                       'linked_list_functions.pyx',
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
                            'stress_funcs.pyx',
                            'linalg.pyx',
                            'gsph_funcs.pyx',
                            'euler1d.pyx',
                            'test_funcs.pyx',
                            'common.pyx',
                            ],
              'solver': ['particle_generator.pyx',
                         ],
              }

parallel_extensions = {'parallel': ['parallel_controller.pyx',
                                    'parallel_cell.pyx',
                                    'parallel_manager.pyx',
                                    ],
                       }

def gen_extensions(ext):
    """Given a dictionary with key package name and value a list of Cython
    files, return a list of Extension instances."""
    modules = []
    for subpkg, files in ext.iteritems():
        for filename in files:
            base = os.path.splitext(filename)[0]
            module = 'pysph.%s.%s'%(subpkg, base)
            module = module.replace("/", ".")
            ext = 'pyx'
            if not HAS_CYTHON:
                ext = C_EXTN
            src = 'source/pysph/%s/%s.%s'%(subpkg, base, ext)
            modules.append(Extension(module, [src]))
    return modules

ext_modules = gen_extensions(extensions)
par_modules = gen_extensions(parallel_extensions)


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
    if HAS_CYTHON and platform.system() != "Windows":
        ext_modules = cythonize(ext_modules,nthreads=ncpu,include_path=inc_dirs)

setup(name='PySPH',
      version = '0.9beta',
      author = 'PySPH Developers',
      author_email = 'pysph-dev@googlegroups.com',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      url = 'http://pysph.googlecode.com',
      license = "BSD",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",
      packages = find_packages('source'),
      package_dir = {'': 'source'},
      
      ext_modules = ext_modules,
      
      include_package_data = True,
      cmdclass=cmdclass,
      #install_requires=['mpi4py>=1.2', 'numpy>=1.0.3', 'Cython>=0.14'],
      #setup_requires=['Cython>=0.14', 'setuptools>=0.6c1'],
      #extras_require={'3D': 'Mayavi>=3.0'},
      zip_safe = False,
      entry_points = """
          [console_scripts]
          pysph_viewer = pysph.tools.mayavi_viewer:main
          """,
      platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
      classifiers = [c.strip() for c in """\
        Development Status :: 4 - Beta
        Environment :: Console
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Natural Language :: English
        Operating System :: MacOS :: MacOS X
        Operating System :: Microsoft :: Windows
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: Python
        Topic :: Scientific/Engineering
        Topic :: Scientific/Engineering :: Physics
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.split()) > 0],
      )

