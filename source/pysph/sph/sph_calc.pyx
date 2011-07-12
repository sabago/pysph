"""
General purpose code for SPH computations.

This module provides the SPHCalc class, which does the actual SPH summation.
    
"""

from libc.stdlib cimport *

cimport numpy
import numpy

from os import path

# logging import
import logging
logger=logging.getLogger()

# local imports
from pysph.base.particle_array cimport ParticleArray, LocalReal, Dummy
from pysph.base.nnps cimport NNPSManager, FixedDestNbrParticleLocator
from pysph.base.nnps cimport NbrParticleLocatorBase

from pysph.sph.sph_func cimport SPHFunction
from pysph.base.carray cimport DoubleArray

from pysph.solver.cl_utils import (HAS_CL, get_cl_include,
    get_pysph_root, cl_read)
if HAS_CL:
    import pyopencl as cl

from pysph.base.locator import LinkedListSPHNeighborLocator

cdef int log_level = logger.level

###############################################################################
# `SPHCalc` class.
###############################################################################
cdef class SPHCalc:
    """ A general purpose summation object
    
    Members:
    --------
    sources -- a list of source particle arrays
    dest -- the destination particles array
    func -- the function to use between the source and destination
    nbr_loc -- a list of neighbor locator for each (source, dest)
    kernel -- the kernel to use
    nnps_manager -- the NNPSManager for the neighbor locator

    Notes:
    ------
    We assume that one particle array is being used in the simulation and
    hence, the source and destination, as well as the func's source and 
    destination are the same particle array.
    
    The particle array should define a flag `bflag` to indicate if a particle
    is a solid of fluid particle. The function can in turn distinguish 
    between these particles and decide whether to include the influence 
    or not.

    The base class and the subclass `SPHFluidSolid` assume that the 
    interaction is to be considered between both types of particles.
    Use the `SPHFluid` and `SPHSolid` subclasses to perform interactions 
    with fluids and solids respectively.
    
    """

    #Defined in the .pxd file
    #cdef public ParticleArray dest
    #cdef public list sources
    #cdef public list funcs
    #cdef public list nbr_locators
    #cdef public NNPSManager nnps_manager
    #cdef public KernelBase kernel
    #cdef public Particles particles
    #cdef public LongArray nbrs

    def __init__(self, particles, list sources, ParticleArray dest,
                  KernelBase kernel, list funcs,
                  list updates=[], integrates=False, dnum=0, nbr_info=True,
                  str id = "", bint kernel_gradient_correction=False,
                  kernel_correction=-1, int dim = 1, str snum="",
                  reset_arrays=True):

        """ Constructor for the SPHCalc object.

        Parameters:
        ------------

        particles : pysph.base.Particles
            The collection of particle arrays used in the simulation.

        dest : pysph.base.ParticleArray
            The particle array on which and SPHOperation is to be evaluated.
            In an equation, the destination particle array will be the LHS
            indexed by the subscript `i` or `a`

        sources : list
            A list of particle arrays that contribute to the destination.
            In the equation representing the SPHOperation, the source
            particle arrays are the RHS indexed by `j` or `b`

        funcs : list
            A list of functions (pysph.sph.sph_func.SPHFunction), one for each
            source that is responsible for computing the interaction between the
            source-destination pair.

        kernel : pysph.base.KernelBase
            The SPH kernel used for the evaluation between the src-dst pair.

        updates : list (default =  [])
            An optional list of strings indicating destination particle array
            properties in which to store the results of the computation.
            By default, the results will be stored in the temporary variables
            `_tmpx` `_tmpy` and `_tmpz` on the destination particle array.
            
        integrates : bool (default = False)
            A flag to indicate that the RHS evaluated by the function is to
            be considered an acceleration which is used for integrating the
            `updates` property when combined with an integrator.

            Example:
            An SPHCalc with updates = ['u', 'v'] and integrates = True will
            cause the integrator (pysph.solver.integrator) to create the
            necessary variables (initial values, step values) which is needed
            for integration.

        dnum : int (1)
            The index of the destination particle array in the list of
            arrays (partilces.arrays).

        nbr_info : bool (True)
            A flag indicating if the evaluation represented by the functions
            requires neighbor information.
            
        id : str ("")
            A unique identification for the Calc. When the calc is constructed
            from the solver interface using operations (pysph.solver.SPHOperation),
            the id is constructed as:

            <dest.name>_<func.id>

        kernel_gradient_correction: bool (False)
            Flag for the bonnet and Lok kernel gradient correction

        kernel_correction : int (-1)
            Integere Identifier for the kind of kernel correction.

        dim : int (1)
            The dimension of the problem. The kernel dim and dim should match

        reset_arrays : bool (False)
            Flag indicating if the output arrays are to be reset.

            For integrating calcs, the integrator is responsible for resetting
            the acceleration arrays to 0 before performing the evaluation.

            For non-integrating calcs, the updates (LHS) list should
            be set to zero before calling the individual functions.

            For example, if we are calculating the density due to summation
            on a fluid from itself and a nearby solid,

            updates = ['rho']
            funcs = [SPHRho(dest=fluid, src=fluid), SPHRho(dest=fluid, src=solid)]

            In this case, we want the density to be set to zero before
            each function is called so that the result they append to
            (dest.rho) will not be affected by any previous value.            

        """
        self.particles = particles
        self.dest = dest

        self.sources = sources
        self.nsrcs = len(sources)

        self.nbr_info = nbr_info

        self.funcs = funcs
        self.kernel = kernel

        self.integrates = integrates

        self.dim = dim

        self.updates = updates
        self.updates = funcs[0].setup_calc_updates(self)
        self.nupdates = len(self.updates)

        self.kernel_correction = kernel_correction

        self.dnum = dnum
        self.id = id

        self.snum = snum
        self.reset_arrays = reset_arrays

        self.correction_manager = None

        self.tag = ""

        self.src_reads = []
        self.dst_reads = []
        self.initial_props = []
        self.dst_writes = {}

        self.context = object()
        self.queue = object()
        self.cl_kernel = object()
        self.cl_kernel_src_file = ''
        self.cl_kernel_function_name = ''

        self.check_internals()
        self.setup_internals()
        
    cpdef check_internals(self):
        """ Check for inconsistencies and set the neighbor locator. """

        # check if the data is sane.

        logger.info("SPHCalc:check_internals: calc %s"%(self.id))

        if (len(self.sources) == 0 or self.nnps_manager is None
            or self.dest is None or self.kernel is None or len(self.funcs)
            == 0):
            logger.warn('invalid input to setup_internals')
            logger.info('sources : %s'%(self.sources))
            logger.info('nnps_manager : %s'%(self.nnps_manager))
            logger.info('dest : %s'%(self.dest))
            logger.info('kernel : %s'%(self.kernel))
            logger.info('sph_funcs : %s'%(self.funcs)) 
            return

        # we need one sph_func for each source.
        if len(self.funcs) != len(self.sources):
            msg = 'One sph function is needed per source'
            raise ValueError, msg

        # ensure that all the funcs are of the same class and have same tag
        funcs = self.funcs
        for i in range(len(self.funcs)-1):
            if type(funcs[i]) != type(funcs[i+1]):
                msg = 'All sph_funcs should be of same type'
                raise ValueError, msg
            if funcs[i].tag != funcs[i+1].tag:
                msg = "All functions should have the same tag"
                raise ValueError, msg

        #check that the function src and dsts are the same as the calc's
        for i in range(len(self.funcs)):
            if funcs[i].source != self.sources[i]:
                msg = 'SPHFunction.source not same as'
                msg += ' SPHCalc.sources[%d]'%(i)
                raise ValueError, msg

            # not valid for SPHFunction
            #if funcs[i].dest != self.dest:
            #    msg = 'SPHFunction.dest not same as'
            #    msg += ' SPHCalc.dest'
            #    raise ValueError, msg

    cdef setup_internals(self):
        """ Set the update update arrays and neighbor locators """

        cdef FixedDestNbrParticleLocator loc
        cdef SPHFunction func
        cdef ParticleArray src
        cdef int i

        func = self.funcs[0]

        # the tag is the same as the function
        self.tag = func.tag

        # set the cl_kernel_function_name
        self.cl_kernel_function_name = func.cl_kernel_function_name

        # a list of properties to reset before calling the functions
        self.to_reset = func.to_reset

        # a list of reads for the calc
        self.src_reads = func.src_reads
        self.dst_reads = func.dst_reads

        # setup the neighbor locators for the functions
        self.nnps_manager = self.particles.nnps_manager
        self.nbr_locators = []
        for i in range(self.nsrcs):
            src = self.sources[i]
            func = self.funcs[i]

            # set the SPHFunction kernel
            func.kernel = self.kernel

            loc = self.nnps_manager.get_neighbor_particle_locator(
                src, self.dest, self.kernel.radius())
            func.nbr_locator = loc

            logger.info("""SPHCalc:setup_internals: calc %s using 
                        locator (src: %s) (dst: %s) %s """
                        %(self.id, src.name, self.dest.name, loc))
            
            self.nbr_locators.append(loc)

    cpdef sph(self, str output_array1=None, str output_array2=None, 
              str output_array3=None, bint exclude_self=False): 
        """
        """
        if output_array1 is None: output_array1 = '_tmpx'
        if output_array2 is None: output_array2 = '_tmpy'
        if output_array3 is None: output_array3 = '_tmpz'

        cdef DoubleArray output1 = self.dest.get_carray(output_array1)
        cdef DoubleArray output2 = self.dest.get_carray(output_array2)
        cdef DoubleArray output3 = self.dest.get_carray(output_array3)

        # reset output arays for non integrating calcs
        if not self.integrates:
            self.reset_output_arrays(output1, output2, output3)

        # reset any other destination properties
        self.dest.set_to_zero(self.to_reset)

        self.sph_array(output1, output2, output3, exclude_self)

        # call an update on the particles if the destination pa is dirty

        if self.dest.is_dirty:
            self.particles.update()

    cpdef sph_array(self, DoubleArray output1, DoubleArray output2, DoubleArray
                     output3, bint exclude_self=False):
        """
        Similar to the sph1 function, except that this can handle
        SPHFunction that compute 3 output fields.

        **Parameters**
        
         - output1 - the array to store the first output component.
         - output2 - the array to store the second output component.
         - output3 - the array to store the third output component.
         - exclude_self - indicates if each particle itself should be left out
           of the computations.

        """

        cdef SPHFunction func

        if self.kernel_correction != -1 and self.nbr_info:
            self.correction_manager.set_correction_terms(self)
        
        for func in self.funcs:
            func.nbr_locator = self.nnps_manager.get_neighbor_particle_locator(
                func.source, self.dest, self.kernel.radius())

            func.eval(self.kernel, output1, output2, output3)

    cpdef tensor_sph(self, str out1=None, str out2=None, str out3=None,
                     str out4=None, str out5=None, str out6=None,
                     str out7=None, str out8=None, str out9=None):
        """Evaluate upto 9 components of a tensor.

        This function is needed to evaluate the nine components of a
        tensor matrix, when used as an integrating calc.

        Example:

        The deviatoric component of the stress tensor is integrated as:

        .. math::

        \frac{dS^{ij}}{dt} = 2\mu\left(\eps^{ij} -
        \frac{1}{3}\delta^{ij}\eps^{ij}\right) + S^{ik}\Omega^{jk} +
        \Omega^{ik}S^{kj}

        The integrator needs to call this function with the
        corresponding acceleration variables for each component.

        Default values _tmpx, _tmpy and _tmpz are chosen for each
        redundant evaluation. By redundant we mean that the 2D version
        of the above function (HookesDeviatoricStressRate2D) will only
        update S_00, S_01, S_10 and S_11. We don't care about the
        rest which by default will be the temporary variables.

        """
        if out1 is None: out1 = "_tmpx"
        if out2 is None: out2 = "_tmpy"
        if out3 is None: out3 = "_tmpz"

        if out1 is None: out4 = "_tmpx"
        if out2 is None: out5 = "_tmpy"
        if out3 is None: out6 = "_tmpz"

        if out1 is None: out7 = "_tmpx"
        if out2 is None: out8 = "_tmpy"
        if out3 is None: out9 = "_tmpz"

        cdef DoubleArray output1 = self.dest.get_carray(out1)
        cdef DoubleArray output2 = self.dest.get_carray(out2)
        cdef DoubleArray output3 = self.dest.get_carray(out3)
        cdef DoubleArray output4 = self.dest.get_carray(out4)
        cdef DoubleArray output5 = self.dest.get_carray(out5)
        cdef DoubleArray output6 = self.dest.get_carray(out6)
        cdef DoubleArray output7 = self.dest.get_carray(out7)
        cdef DoubleArray output8 = self.dest.get_carray(out8)
        cdef DoubleArray output9 = self.dest.get_carray(out9)

        # call the tensor eval function
        self.tensor_sph_array(output1, output2, output3,
                              output4, output5, output6,
                              output7, output8, output9)

    cpdef tensor_sph_array(self, DoubleArray output1, DoubleArray output2,
                           DoubleArray output3, DoubleArray output4,
                           DoubleArray output5, DoubleArray output6,
                           DoubleArray output7, DoubleArray output8,
                           DoubleArray output9):

        cdef SPHFunction func
        for func in self.funcs:
            func.nbr_locator = self.nnps_manager.get_neighbor_particle_locator(
                func.source, self.dest, self.kernel.radius())

            func.tensor_eval(self.kernel,
                             output1, output2, output3,
                             output4, output5, output6,
                             output7, output8, output9)

    cdef reset_output_arrays(self, DoubleArray output1, DoubleArray output2,
                             DoubleArray output3):
        if not self.reset_arrays:
            return

        cdef int i
        for i in range(output1.length):
            output1.data[i] = 0.0
            output2.data[i] = 0.0
            output3.data[i] = 0.0

#############################################################################

class CLCalc(SPHCalc):
    """ OpenCL aware SPHCalc """

    def __init__(self, particles, sources, dest, kernel, funcs,
                 updates, integrates=False, dnum=0, nbr_info=True,
                 str id = "", bint kernel_gradient_correction=False,
                 kernel_correction=-1, int dim = 1, str snum=""):

        self.nbr_info = nbr_info
        self.particles = particles
        self.sources = sources
        self.nsrcs = len(sources)
        self.dest = dest

        self.funcs = funcs
        self.kernel = kernel

        self.integrates = integrates
        self.updates = updates
        self.nupdates = len(updates)

        self.dnum = dnum
        self.id = id

        self.dim = dim
        self.snum = snum

        self.tag = ""

        self.src_reads = []
        self.dst_reads = []
        self.initial_props = []
        self.dst_writes = {}

        self.context = object()
        self.queue = object()
        self.cl_kernel = object()
        self.cl_kernel_src_file = ''
        self.cl_kernel_function_name = ''

        self.check_internals()
        self.setup_internals()

    def setup_internals(self):
        self.cl_kernel_function_name = self.funcs[0].cl_kernel_function_name
        self.tag = self.funcs[0].tag

        for func in self.funcs:
            func.kernel = self.kernel

    def setup_cl_kernel_file(self):
        func = self.funcs[0]
        
        src = get_pysph_root()
        src = path.join(src, 'sph/funcs/' + func.cl_kernel_src_file)

        if not path.isfile(src):
            fname = func.cl_kernel_src_file
            logger.debug("OpenCL kernel file does not exist: %s"%fname)
            
        self.cl_kernel_src_file = src

    def setup_cl(self, context):
        """ Setup the CL related stuff

        Parameters:
        -----------

        context -- the OpenCL context to use

        The Calc creates an OpenCL CommandQueue using, by default, the
        first available device within the context.

        The ParticleArrays are allocated on the device by calling
        their respective setup_cl functions.

        I guess we would need a clean way to deallocate the device
        arrays once we are through with one step and move on to the
        next.

        """
        self.setup_cl_kernel_file()
        
        self.context = context
        self.devices = context.devices

        # create a command queue with the first device on the context 
        self.queue = cl.CommandQueue(context, self.devices[0])

        # set up the domain manager and device buffers for the srcs and dest
        self.particles.setup_cl(self.context)

        # setup the locators on each function
        self.set_cl_locator()

        self.setup_program()

    def _create_program(self, template):
        """Given a program source template, flesh out the code and
        return it.

        An OpenCL kernel template requires the following pieces of
        code to make it a valid OpenCL kerel file:

        kernel_args -- the arguments for the kernel injected by the
                       function.

        workgroup_code -- code injected by the function to get the
                          global id from the kernel launch parameters.

        neighbor_loop_code -- code injected by the neighbor locator to
                              determine the next neighbor for a given
                              particle                       

        """
        k_args = []

        for func in self.funcs:
            func.set_cl_kernel_args()

        func = self.funcs[0]
        k_args.extend(func.cl_args_name)

        # Build the kernel args string.
        kernel_args = ',\n    '.join(k_args)

        # Get the kernel workgroup code
        workgroup_code = func.get_cl_workgroup_code()

        # get the neighbor loop code
        locator = func.cl_locator

        neighbor_loop_code_start = locator.neighbor_loop_code_start()
        neighbor_loop_code_end = locator.neighbor_loop_code_end()
        neighbor_loop_code_break = locator.neighbor_loop_code_break()

        return template%(locals())

    def setup_program(self):
        """ Setup the OpenCL function used by this Calc
        
        The calc computes the interaction on a single destination from
        a list of sources, using the same function.

        A call to this function sets up the OpenCL kernel which
        encapsulates the SPHFunction function.

        The OpenCL source file template is read and suitable code is
        injected to make it work.

        An example template file is 'sph/funcs/density_funcs.cl'

        """

        prog_src_tmp = cl_read(self.cl_kernel_src_file,
                               precision=self.particles.get_cl_precision(),
                               function_name=self.cl_kernel_function_name)

        prog_src = self._create_program(prog_src_tmp)
        
        self.cl_kernel_src = prog_src

        build_options = get_cl_include()

        self.prog = cl.Program(self.context, prog_src).build(
            build_options)

        # set the OpenCL kernel for each of the SPHFunctions
        for func in self.funcs:
            func.setup_cl(self.prog, self.context)

    def set_cl_locator(self):
        dst = self.dest
        manager = self.particles.domain_manager

        if self.nsrcs == 0:
            self.funcs[0].set_cl_locator(
                self.particles.get_neighbor_locator(dst, dst)
                )
        else:    
            for i in range(self.nsrcs):
                func = self.funcs[i]
                src = self.sources[i]

                func.set_cl_locator(
                    self.particles.get_neighbor_locator(src, dst)
                    )

    def sph(self, output1=None, output2=None, output3=None):
        """ Evaluate the contribution from the sources on the
        destinations using OpenCL.

        Each of the ParticleArray's properties has an equivalent device
        representation. The devie buffers have the suffix `cl_` so that
        a numpy array `rho` on the host has the buffer `cl_rho` on the device.

        The device buffers are created upon a call to CLCalc's setup_cl
        function. The CommandQueue is created using the first device
        in the context as default. This could be changed in later
        revisions.
        
        Each of the device buffers can be obtained like so:

        pa.get_cl_buffer(prop)

        where prop may not be suffixed with 'cl_'
        
        The OpenCL kernel functions have the signature:

        __kernel void function(__global int* kernel_type, __global int* dim,
                               __global int* nbrs,
                               __global float* d_x, __global float* d_y,
                               __global_float* d_z, __global float* d_h,
                               __global int* d_tag,
                               __global float* s_x, __global float* s_y,
                               ...,
                               __global float* tmpx, ...)

        Thus, the first three arguments are the sph kernel type,
        dimension and a list of neighbors for the destination particle.

        The following properties are the destination particle array
        properties that are required. This is defined in each function
        as the attribute dst_reads

        The last arguments are the source particle array properties
        that are required. This i defined in each function as the
        attribute src_reads.

        The final arguments are the output arrays to which we want to
        append the result. The number of the output arguments can vary
        based on the function.
    
        """
        
        # set the tmpx, tmpy and tmpz arrays for dest to 0

        if output1 is None: output1 = '_tmpx'
        if output2 is None: output2 = '_tmpy'
        if output3 is None: output3 = '_tmpz'

        if not self.integrates:
            self.cl_reset_output_arrays(output1, output2, output3)

        for func in self.funcs:
            func.cl_eval(self.queue, self.context, output1, output2, output3)

    def cl_reset_output_arrays(self, output1, output2, output3):
        """ Reset the dst tmpx, tmpy and tmpz arrays to 0

        Since multiple functions contribute to the same LHS value, the
        OpenCL kernel code increments the result to tmpx, tmpy and tmpz.
        To avoid unplesant behavior, we set these variables to 0 before
        any function is called.
        
        """
        if not self.reset_arrays:
            return

        dest = self.dest

        cl_tmpx = dest.get_cl_buffer(output1)
        cl_tmpy = dest.get_cl_buffer(output2)
        cl_tmpz = dest.get_cl_buffer(output3)

        npd = self.dest.get_number_of_particles()

        self.prog.set_tmp_to_zero(self.queue, (npd, 1, 1), (1, 1, 1),
                                  cl_tmpx, cl_tmpy, cl_tmpz).wait()

#############################################################################
