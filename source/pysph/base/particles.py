from cell import CellManager
from nnps import NNPSManager, NeighborLocatorType
from particle_array import ParticleArray
from particle_types import ParticleType

from domain_manager import DomainManagerType as CLDomain
from locator import OpenCLNeighborLocatorType as CLLocator
import locator

import domain_manager

from pysph.solver.cl_utils import HAS_CL

if HAS_CL:
    import pyopencl as cl

Fluid = ParticleType.Fluid
Solid = ParticleType.Solid
Probe = ParticleType.Probe
DummyFluid = ParticleType.DummyFluid
Boundary = ParticleType.Boundary

SPHNeighborLocator = NeighborLocatorType.SPHNeighborLocator

# MPI conditional imports
HAS_MPI = True
try:
    from mpi4py import MPI
except ImportError:
    HAS_MPI = False
else:
    from pysph.parallel.parallel_cell import ParallelCellManager

import numpy

class Particles(object):
    """ A collection of particles and related data structures that
    hat define an SPH simulation.

    In pysph, particle properties are stored in a ParticleArray. The
    array may represent a particular type of particle (solid, fluid
    etc). Valid types are defined in base.particle_types.

    Indexing of the particles is performed by a CellManager and
    nearest neighbors are obtained via an instance of NNPSManager.

    Particles is a collection of these data structures to provide a
    single point access to

    (a) Hold all particle information
    (b) Update the indexing scheme when particles have moved.
    (d) Update remote particle properties in parallel runs.
    (e) Barrier synchronizations across processors

    Data Attributes:
    ----------------

    arrays -- a list of particle arrays in the simulation.

    cell_manager -- the CellManager for spatial indexing.

    nnps_manager -- the NNPSManager for neighbor queries.

    correction_manager -- a kernel KernelCorrectionManager if kernel
    correction is used. Defaults to None

    misc_prop_update_functions -- A list of functions to evaluate
    properties at the beginning of a sub step.

    variable_h -- boolean indicating if variable smoothing lengths are
    considered. Defaults to False

    in_parallel -- boolean indicating if running in parallel. Defaults to False

    load_balancing -- boolean indicating if load balancing is required.
    Defaults to False.

    pid -- processor id if running in parallel

    Example:
    ---------

    In [1]: import pysph.base.api as base

    In [2]: x = linspace(-pi,pi,101)
    
    In [3]: pa = base.get_particle_array(x=x)
    
    In [4]: particles = base.Particles(arrays=[pa], in_parallel=True,
                                       load_balancing=False, variable_h=True)


    Notes:
    ------

    An appropriate cell manager (CellManager/ParallelCellManager) is
    created with reference to the 'in_parallel' attribute.

    Similarly an appropriate NNPSManager is created with reference to
    the 'variable_h' attribute.

    """
    
    def __init__(self, arrays=[], in_parallel=False, variable_h=False,
                 load_balancing=True,
                 locator_type = SPHNeighborLocator,
                 periodic_domain=None,
                 min_cell_size=-1,
                 max_radius_scale=2,
                 update_particles=True):
        
        """ Constructor

        Parameters:
        -----------

        arrays -- list of particle arrays in the simulation

        in_parallel -- flag for parallel runs

        variable_h -- flag for variable smoothing lengths

        load_balancing -- flag for dynamic load balancing.

        periodic_domain -- the periodic domain for periodicity

        """

        # set the flags

        self.variable_h = variable_h
        self.in_parallel = in_parallel
        self.load_balancing = load_balancing
        self.locator_type = locator_type
        self.min_cell_size = min_cell_size
        self.periodic_domain = periodic_domain
        self.parallel_manager = None

        self.max_radius_scale = max_radius_scale

        # Some sanity checks on the input arrays.
        assert len(arrays) > 0, "Particles must be given some arrays!"
        prec = arrays[0].cl_precision
        msg = "All arrays must have the same cl_precision"
        for arr in arrays[1:]:
            assert arr.cl_precision == prec, msg

        self.arrays = arrays
        self.kernel = None

        # set defaults
        self.correction_manager = None        
        self.misc_prop_update_functions = []

        # initialize the cell manager and nnps manager
        self.initialize()

    def initialize(self):
        """ Perform all initialization tasks here """

        # create the cell manager
        #if not self.in_parallel:
        self.cell_manager = CellManager(arrays_to_bin=self.arrays,
                                        min_cell_size=self.min_cell_size,
                                        max_radius_scale=self.max_radius_scale,
                                        periodic_domain=self.periodic_domain)
        #else:
        #    self.cell_manager = ParallelCellManager(
        #        arrays_to_bin=self.arrays, load_balancing=self.load_balancing)

        #self.pid = self.cell_manager.pid

        # create the nnps manager
        self.nnps_manager = NNPSManager(cell_manager=self.cell_manager,
                                        variable_h=self.variable_h,
                                        locator_type=self.locator_type)

        # call an update on the particles (i.e index)
        self.update()

    def update(self, cache_neighbors=False):
        """ Update the status of the Particles.

        Parameters:
        -----------

        cache_neighbors -- flag for caching kernel interactions 
        
        Notes:
        -------
        
        This function must be called whenever particles have moved and
        the indexing structure invalid. After a call to this function,
        particle neighbors will be accurately returned. 

        Since particles move at the end of an integration
        step/sub-step, we may perform any other operation that would
        be required for the subsequent step/sub-step. Examples of
        these are summation density, equation of state, smoothing
        length updates, evaluation of velocity divergence/vorticity
        etc. 

        All other properties may be updated by appending functions to
        the list 'misc_prop_update_functions'. These functions must
        implement an 'eval' method which takes no arguments. An example
        is the UpdateDivergence function in 'sph.update_misc_props.py'
        
        """

        pm = self.parallel_manager
        if pm is not None:
            pm.update()

        err = self.nnps_manager.py_update()
        assert err != -1, 'NNPSManager update failed! '            

        # update any other properties (rho, p, cs, div etc.)
            
        self.evaluate_misc_properties()

        # evaluate kernel correction terms

        if self.correction_manager:
            self.correction_manager.update()

    def evaluate_misc_properties(self):
        """ Evaluate properties from the list of functions. """
        
        for func in self.misc_prop_update_functions:
            func.eval()

    def add_misc_function(self, func):
        """ Add a function to be performed when particles are updated

        Parameters:
        -----------
        func -- The function to perform.

        Example:
        --------

        The conduction coefficient required for the artificial heat
        requires the velocity divergence at a particle. This must be
        available at the start of every substep of an integration step.

        """

        #calcs = operation.get_calcs(self, kernel)
        self.misc_prop_update_functions.append(func)

    def get_named_particle_array(self, name):
        """ Return the named particle array if it exists """
        has_array = False

        for array in self.arrays:
            if array.name == name:
                arr = array
                has_array = True                
                
        if has_array:
            return arr
        else:
            print 'Array %s does not exist!' %(name)

    def update_remote_particle_properties(self, props=None):
        """ Perform a remote particle property update. 
        
        This function needs to be called when the remote particles
        on one processor need to be updated on account of computations 
        on another physical processor.

        """
        if self.in_parallel:
            self.parallel_manager.update_remote_particle_properties(props=props)

    def barrier(self):
        """ Synchronize all processes """
        if self.in_parallel:
            self.parallel_manager.parallel_controller.comm.barrier()

    def get_global_min_max(self, props):
        """ Find the global minimum and maximum values.

        Parameters:
        -----------

        props : dict
            A dict of local properties for which we want global values.

        """

        data_min = {}
        data_max = {}

        for prop in props:
            data_min[prop] = props[prop]
            data_max[prop] = props[prop]

        pc = self.parallel_manager.parallel_controller
        glb_min, glb_max = pc.get_glb_min_max(data_min, data_max)

        return glb_min, glb_max

    @classmethod
    def get_neighbor_particle_locator(self, src, dst, 
                                      locator_type = SPHNeighborLocator,
                                      variable_h=False, radius_scale=2.0):
        """ Return a neighbor locator from the NNPSManager """

        cell_manager = CellManager(arrays_to_bin=[src, dst])
        nnps_manager = NNPSManager(cell_manager, locator_type=locator_type,
                                   variable_h=variable_h)

        return nnps_manager.get_neighbor_particle_locator(
            src, dst, radius_scale)

class CLParticles(Particles):
    """ A collection of ParticleArrays for use with OpenCL.

    CLParticles is modelled very closely on `Particles` which is
    intended for Cython computations.

    Use CLParticles when using a CLCalc with OpenCL.

    Attributes:
    -----------

    arrays : list
        The list of arrays considered in the solution

    with_cl : bool {True}
         Duh
         
    domain_manager_type : int base.DomainManagerType
        A domain manager is used to spatially index the particles and provide
        an interface which is accesible and comprehensible to an appropriate
        OpenCLNeighborLocator object.

        Acceptable values are:
        (1) base.DomainManagerType.LinkedListManager : Indexing based on the
            linked list structure defined by Hockney and Eastwood.

        (2) base.DomainManagerType.DomainManager : No indexing. Intended to
            be used for all pair neighbor searches.

    cl_locator_type : int base.OpenCLNeighborLocatorType
        A neighbor locator is in cahoots with the DomainManager to provide
        near neighbors for a particle upon a query.

        Acceptable values are:
        (1) base.OpenCLNeighborLocatorType.LinkedListSPHNeighborLocator :
           A neighbor locator that uses the linked list structure of the
           LinkedListManager to provide neighbors in an SPH context. That
           is, nearest neighbors are particles in the 27 neighboring cells
           for the destination particle.

        (2) base.OpenCLNeighborLocatorType.AllPairNeighborLocator :
            A trivial locator that essentially returns all source particles
            as near neighbors for any query point.

    """
    def __init__(self, arrays,
                 domain_manager_type=CLDomain.DomainManager,
                 cl_locator_type=CLLocator.AllPairNeighborLocator):

        self.arrays = arrays
        self.with_cl = True

        self.domain_manager_type = domain_manager_type
        self.cl_locator_type = cl_locator_type

        self.in_parallel = False

    def get_cl_precision(self):
        """Return the cl_precision used by the Particle Arrays.

        This property cannot be set it is set at construction time for
        the Particle arrays.  This is simply a convenience function to
        query the cl_precision.
        """
        # ensure that all arrays have the same precision
        narrays = len(self.arrays)
        if ( narrays > 1 ):
            for i in range(1, narrays):
                assert self.arrays[i-1].cl_precision == \
                       self.arrays[i].cl_precision

        return self.arrays[0].cl_precision

    def setup_cl(self, context):
        """ OpenCL setup given a context.

        Parameters:
        -----------

        context : pyopencl.Context

        The context is used to instantiate the domain manager, the
        type of which is determined from the attribute
        `domain_manager_type`.

        I expect this function to be called from the associated
        CLCalc, from within it's `setup_cl` method.  The point is that
        the same context is used for the Calc, the DomainManager and
        the underlying ParticleArrays. This is important as a mix of
        contexts will result in crashes.

        The DomainManager is updated after creation. This means that
        the data is ready to be used by the SPHFunction OpenCL
        kernels.

        """
        self.with_cl = True
        self.context = context

        # create the domain manager.
        self.domain_manager = self.get_domain_manager(context)

        # Update the domain manager
        self.domain_manager.update_status()
        self.domain_manager.update()

    def get_domain_manager(self, context):
        """ Get the domain manager from type. """ 

        if self.domain_manager_type == CLDomain.DomainManager:
            return  domain_manager.DomainManager(
                arrays = self.arrays, context = context
                )

        if self.domain_manager_type == CLDomain.LinkedListManager:
            return domain_manager.LinkedListManager(
                arrays=self.arrays, context = context
                )

        else:
            msg = "Manager type %s not understood!"%(self.domain_manager_type)
            raise ValueError(msg)

    def get_neighbor_locator(self, source, dest, scale_fac=2.0):
        """ Return an OpenCLNeighborLocator between a source and
        destination.

        Parameters:
        -----------

        source : ParticleArray
            The source particle array

        dest : ParticleArray
            The destination particle array

        scale_fac : float
            NOTIMPLEMENTED. The scale facor to determine the effective
            cutoff radius.

        Note:
        -----

        An error is raised if a linked list neighbor locator is
        requested with a domain manager other than the
        LinkedListManager. 

        """
        if self.cl_locator_type == \
               CLLocator.AllPairNeighborLocator:

            return locator.AllPairNeighborLocator(source=source, dest=dest)

        if self.cl_locator_type == \
               CLLocator.LinkedListSPHNeighborLocator:

            if not self.domain_manager_type == \
                   CLDomain.LinkedListManager:
                raise RuntimeError

            return locator.LinkedListSPHNeighborLocator(
                manager=self.domain_manager, source=source, dest=dest,
                scale_fac=scale_fac)

    def update(self):
        """ Update the spatial index of the particles.

        First check if the domain manager needs an update by calling
        it's update_status method and then proceed with the update.

        The reason this is done is to avoid any repeated updates.

        """
        self.domain_manager.update_status()
        self.domain_manager.update()

    def read_from_buffer(self):
        """ Read the buffer contents for all the arrays """
        for pa in self.arrays:
            pa.read_from_buffer()

###############################################################################
