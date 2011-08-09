"""API module to simplify import of common names from pysph.base package"""

# fast utils
from fast_utils import arange_long

# carray
from carray import LongArray, DoubleArray, IntArray, FloatArray

from cell import Cell, CellManager, PeriodicDomain

from kernels import KernelBase, DummyKernel, CubicSplineKernel, \
        HarmonicKernel, GaussianKernel, M6SplineKernel, W8Kernel, W10Kernel,\
        QuinticSplineKernel, WendlandQuinticSplineKernel, Poly6Kernel

from nnps import NbrParticleLocatorBase, FixedDestNbrParticleLocator, \
        VarHNbrParticleLocator, NNPSManager, brute_force_nnps

from nnps import NeighborLocatorType

from particle_array import ParticleArray, get_particle_array
from particles import Particles, CLParticles

from point import Point, IntPoint

# ParticleTypes
from particle_types import ParticleType
Fluid = ParticleType.Fluid
Solid = ParticleType.Solid
Boundary = ParticleType.Boundary
Probe = ParticleType.Probe
DummyFluid = ParticleType.DummyFluid

from geometry import MeshPoint, Line, Geometry

# LinkedListManager
from domain_manager import LinkedListManager, DomainManager, \
     DomainManagerType

# OpenCL locator
from locator import OpenCLNeighborLocator, LinkedListSPHNeighborLocator, \
     AllPairNeighborLocator

from locator import OpenCLNeighborLocatorType

