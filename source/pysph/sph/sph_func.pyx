#include for malloc
from libc.stdlib cimport * 
cimport numpy
import numpy

from pysph.solver.cl_utils import get_real

def get_all_funcs():
    ''' function to gather all implemented funcs in pysph.sph.funcs package '''
    import os
    import pysph.sph.funcs as funcs_pkg
    funcs = {}
    for funcs_dir in funcs_pkg.__path__:
        search_modules = [os.path.splitext(i) for i in os.listdir(funcs_dir)]
        search_modules = [i[0] for i in search_modules if i[1]=='.pyx' and i[0][0]!='.']
        for mod_name in search_modules:
            mod_name = 'pysph.sph.funcs.'+mod_name
            mod = __import__(mod_name, fromlist=True)
            for name,value in mod.__dict__.iteritems():
                if type(value) == type and issubclass(value, SPHFunction) and (
                        not name.startswith('SPHFunction')):
                    funcs['%s.%s'%(mod_name,name)] = value
    return funcs


class Function(object):
    """ Class that defines sph function (sph.funcs) and its parameter values

    **Methods**

    - get_func -- Return a particular instance of SPHFunctionParticle
      with an appropriate source and destination particle array
    
    - get_func_class --  get the class for which func will be created
    
    **Example**

    The sph function MonaghanArtificialVsicosity (defined in
    sph.funcs.viscosity_funcs) requires the parameter values 'alpha',
    'beta', 'gamma' and 'eta' to define the artificial viscosity. This
    function may be created as:

    - avisc = Function(MonaghanArtificialVsicosity, hks=False, alpha, beta ..)
    - avisc_func = avisc.get_funcs(source, dest)
    
    or as an alternative may also be created as follows:
    
    - avisc = MonaghanArtificialVsicosity.withargs(hks=False, alpha, beta, ..)
    - avisc_func = avisc.get_funcs(source, dest)
    
    Function provides a convenient way to create funcs between multiple
    source and destination particle arrays with specified
    parameter values

    """
    def __init__(self, sph_func, *args, **kwargs):
        """ Constructor

        Parameters:
        -----------

        sph_func -- the SPHFunction class type
        *args, **kwargs -- optional positional and keyword arguments

        """

        self.sph_func = sph_func
        self.args = args
        self.kwargs = kwargs
    
    def get_func_class(self):
        """ Get the class for which func will be created """
        return self.sph_func
    
    def get_func(self, source, dest):
        """ Return a SPHFunctionParticle instance with source and dest """
        func = self.sph_func(source, dest, *self.args, **self.kwargs)
        return func

################################################################################
# `SPHFunction` class.
################################################################################
cdef class SPHFunction:
    """ Base class to represent an operation on a particle array.

    This class requires access to particle properties of a ParticleArray.
    Since there is no particle class having all properties at one place,
    names used for various properties and arrays corresponding to those
    properties are stored in this class for fast access to property values both
    at the source and destination.
    
    This class contains names, and arrays of common properties that will be
    needed for an operation. The data within
    these arrays, can be used as ``array.data[pid]``, where ``pid`` in the particle
    index, ``data`` is the actual c-pointer to the data.

    All arrays are prefixed with a ``s_``. Destination arrays prefixed
    by ``d_`` are an alias for the same array prefixed with ``s_``. For
    example the mass property of the source will be in the ``s_m`` array
    which is same as ``d_m``.  This is not true for subclasses of
    :class:`SPHFunctionParticle` which can have different source and
    destination pairs

    """
    def __init__(self, list arrays, list on_types, object dm,
                 object function_data):

        """Constructor.

        The SPHFunction object is a generic 'function' that operates
        on a bunch of destination particle arrays. The
        source-destination pairs for the interactions are decided at
        run time from the on types and from types arguments. The
        kernel is used to compute the interaction if necessary. 

        """
        self.name = ""
        self.id = ""
        self.tag = ""

        self.arrays = arrays
        self.on_types = on_types

        self.src_reads = []
        self.dst_reads = []

        # sph kernel used to evaluate contributions
        self.kernel = None

        # domain manager for the spatial indexing
        self.dm = dm

        # destination arrays
        self.dsts = []

        # destination arrays data
        self.dst_data = []

        # generate the destination particle arrays. The destination
        # arrays are chosen from the on_types specification and the
        # 'dest' object actually stored is a ParticleArrayData which
        # gives access to all requisite carrays for the function.
        for dst in arrays:
            if dst in on_types:
                self.dsts.append( dst )
                self.dst_data.append( function_data(dst) )

        # number of destination arrays
        self.ndsts = len( self.dsts )

        # a list of destination props that may need resetting at the
        # calc level. We can't reset at the function level since the
        # values need to be computed across functions
        self.to_reset = []                

        #################################################################
        # OpenCL support possibly deprecated within the new framework
        # This will have to be re-worked
        #################################################################
        
        self.cl_kernel_src_file = ''
        self.cl_kernel = object()
        self.cl_program = object()
        self.context = object()
        self.cl_locator = object()

        self.cl_args = []
        self.cl_args_name = []

        #self.global_sizes = (self.dest.get_number_of_particles(), 1, 1)
        self.local_sizes = (1,1,1)

        # setup the source and destination reads
        #if setup_reads:
        #    self.set_src_dst_reads()

    cpdef eval(self, double t=0, double dt=0):
        """Evaluate the function.

        SPHFunction calculates some quiantity for destination particle
        arrays without any recourse to neighbor lookups.

        The simplest example for this case is the external gravity
        force that is added to each particle.

        """
        cdef ParticleArray dst
        cdef size_t i, np
        cdef object dm = self.dm

        cdef size_t this_cell

        # loop over all cells
        print "Looping over all cells"
        for nbr_cells in dm:
            this_cell = nbr_cells[0]

            print "This cell is ", this_cell

            # loop over the destination arrays
            for dst_id in range( self.ndsts ):
                dst = self.dsts[dst_id]

                dst_indices = dm.get_indices( dst, this_cell )

                print "Destination = %s, indices = %s"%(dst.name, dst_indices)

                # evaluate the contribution to the destination for this cell
                print "Evaluating on a single cell"
                self.eval_single( dst_id, dst_indices, nbr_cells, t, dt )

    cdef void eval_single(self, size_t dst_id, object dst_indices, object nbr_cells,
                          double t=0, double dt=0):
        """ Evaluate the function on a single dest particle."""
        raise NotImplementedError, 'SPHFunction.eval_single()'
    	
    cpdef setup_iter_data(self):
        """ setup operations performed in each iteration
        
        Override this in a subclass to do any operations at every iteration
        """
        pass

    def setup_calc_updates(self, object calc):
        """ Set the updates list for an the SPHCalc object.

        The default behavior is to return the updates list for the
        calc itself. Tensor functions (especially in 3D) would require
        at times 9 'updates' which could be tedious and error
        prone. This is left to the function which defines the 'updates'
        for the calc and sets the calc's tensor eval flag to True.

        This is only required for integrating calcs when we're
        computing the accelerations for the nine (3D) components of a
        tensor.

        For non integratig calcs that evaluate the components of a
        tensor on the LHS, the SPHFunction's eval method can be
        overrided. Look at VelocityGradient in basic_funcs for an
        example.

        """
        return calc.updates        

    def setup_cl(self, object program, object context):
        """ OpenCL setup for the function.

        You may determine the OpenCL kernel launch parameters from within
        this function

        Currently we're using the default:

        global_sizes = (ndp, 1, 1)
        local_sizes = (1, 1, 1)

        """
        self.cl_program = program
        self.context = context

    def set_cl_kernel_args(self, output1=None, output2=None, output3=None):

        if output1 is None: output1 = '_tmpx'
        if output2 is None: output2 = '_tmpy'
        if output3 is None: output3 = '_tmpz'

        self.cl_args_name = []
        self.cl_args = []
        
        # locator args
        locator_args = self.cl_locator.get_kernel_args()

        for arg_name, arg in locator_args.iteritems():
            self.cl_args.append(arg)
            self.cl_args_name.append(arg_name)

        precision = self.dest.cl_precision

        # kernel args
        if self.kernel is not None:

            kernel_radius = get_real(self.kernel.radius(), precision)
            kernel_type = numpy.int32(self.kernel.get_type())
            dim = numpy.int32(self.kernel.dim)

            self.cl_args.append(kernel_radius)
            self.cl_args_name.append("REAL const kernel_radius")

            self.cl_args.append(kernel_type)
            self.cl_args_name.append('int const kernel_type')

            self.cl_args.append(dim)
            self.cl_args_name.append('int const dim')
        
        for prop in self.dst_reads:
            self.cl_args.append(self.dest.get_cl_buffer(prop))

            if not prop == "tag":
                self.cl_args_name.append('__global REAL* d_%s'%(prop))
            else:
                self.cl_args_name.append('__global int* d_tag')

        for prop in self.src_reads:
            self.cl_args.append(self.source.get_cl_buffer(prop))
            self.cl_args_name.append('__global REAL* s_%s'%(prop))

        # append the output buffer. 
        self.cl_args.append( self.dest.get_cl_buffer(output1) )
        self.cl_args_name.append('__global REAL* tmpx')

        self.cl_args.append( self.dest.get_cl_buffer(output2) )
        self.cl_args_name.append('__global REAL* tmpy')

        self.cl_args.append( self.dest.get_cl_buffer(output3) )
        self.cl_args_name.append('__global REAL* tmpz')

        self._set_extra_cl_args()

    def _set_extra_cl_args(self):
        raise NotImplementedError("SPHFunction _set_extra_cl_args!")
        
    def set_cl_program(self, object program):
        self.cl_program = program

    def set_cl_locator(self, object locator):
        self.cl_locator = locator

    def get_cl_workgroup_code(self):
        return """ unsigned int dest_id = get_global_id(0);"""

    def set_src_dst_reads(self):
        """ Populate the read requirements for the Function

        The read requirements specify which particle properties will
        be required by this function. Properties read from the source
        particle array are appended to the list ``src_reads`` and those
        from the destination particle array are appended to
        ``dst_reads``

        These read requirements are used to construct the OpenCL
        kernel arguments at program creation time.

        """
        raise NotImplementedError("SPHFunction set_src_dst_reads called!")

################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunctionParticle(SPHFunction):
    """ Base class to represent an interaction between two particles from two
    possibly different particle arrays.

    This class requires access to particle properties of possibly two different
    entities. Since there is no particle class having all properties at one
    place, names used for various properties and arrays corresponding to those
    properties are stored in this class for fast access to property values both
    at the source and destination.
    
    This class contains names, and arrays of common properties that will
    be needed for any particle-particle interaction computation. The
    data within these arrays, can be used as ``array.data[pid]``, where
    ``pid`` in the particle index, ``data`` is the actual c-pointer to
    the data.

    All source arrays are prefixed with a ``s_``. All destination arrays
    are prefixed by a ``d_``. For example the mass property of the
    source will be in the ``s_m`` array.

    """
    def __init__(self, list arrays, list on_types, object dm,
                 object function_data, list from_types, KernelBase kernel=None,
                 **kwargs):
        SPHFunction.__init__(self, arrays, on_types, dm, function_data)

        #kernel correction of Bonnet and Lok
        self.bonnet_and_lok_correction = False

        #flag for the rkpm first order kernel correction
        self.rkpm_first_order_correction = False

        # type of kernel symmetrization
        self.hks = True

        # SPH kernel to use
        self.kernel = kernel

        # from types to determine the sources
        self.from_types = from_types

        # source particle arrays
        self.srcs = []

        # source particles array data
        self.src_data = []

        # generate the source particle arrays. The source
        # arrays are chosen from the on_types specification and the
        # 'source' object actually stored is a ParticleArrayData which
        # gives access to all requisite carrays for the function.
        for pa in arrays:
            if pa in from_types:
                self.srcs.append( pa )
                self.src_data.append( function_data(pa) )

        # number of destination arrays
        self.nsrcs = len( self.srcs )

    cdef void eval_single(self, size_t dst_id, object dst_indices, object nbr_cells,
                          double t=0, double dt=0):
        """ Computes contribution of all neighbors on particle at dest_pid """
        cdef ParticleArray src
        cdef ParticleArray dst = self.dsts[dst_id]
        cdef size_t src_id

        cdef object dm = self.dm

        this_cell = nbr_cells[0]

        # Handle interactions from within the same cell
        for src_id in range(self.nsrcs):
            src = self.srcs[src_id]
            is_symmetric = src in self.dsts

            # handle interactions for same cell same array
            if dst == src:
                print "Interactions on same cell and same array"
                self.eval_self( dst_id, dst_indices )

            # handle interactions for same cell different array
            else:
                print "Interaction on same cell differnet array"
                src_indices = dm.get_indices( src, this_cell )
                self.eval_nbr( dst_id, src_id,
                               dst_indices, src_indices, is_symmetric )

        # Handle interactions from neighboring cells
        print "Interaction between neighboring cells"
        for cid in nbr_cells[1:]:
            print "Neighboring cell %d"%(cid)
            for src_id in range(self.nsrcs):
                src = self.srcs[src_id]

                is_symmetric = src in self.dsts

                print "Interaction is symmetric? %s "%(is_symmetric)
                
                src_indices = dm.get_indices( src, cid )
                #dst_indices = dm.get_indices( dst, cid )

                print "dst_indices = %s, src_indices = %s"%(dst_indices,src_indices)
                
                self.eval_nbr( dst_id, src_id,
                               dst_indices, src_indices, is_symmetric )
                    
    cdef void eval_self(self, size_t dst_id, object dst_indices):
        raise NotImplementedError, "SPHFunctionParticle.eval_self()"

    cdef void eval_nbr(self, size_t dst_id, size_t src_id,
                       object dst_indices, object src_indices, bint is_symmetric):
        raise NotImplementedError, 'SPHFunctionParticle.eval_nbr()'

    cdef double rkpm_first_order_kernel_correction(self, size_t dest_pid):
        """ Return the first order correction term for an interaction """

        cdef double beta1, beta2, alpha
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        
        beta1 = self.d_beta1.data[dest_pid]
        beta2 = self.d_beta2.data[dest_pid]
        alpha = self.d_alpha.data[dest_pid]

        return alpha * (1.0 + beta1 * rab.x + beta2 * rab.y)

    cdef double rkpm_first_order_gradient_correction(self, size_t dest_pid):
        """ Return the first order correction term for an interaction """
        
        cdef double beta1, beta2, alpha
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        
        beta1 = self.d_beta1.data[dest_pid]
        beta2 = self.d_beta2.data[dest_pid]
        alpha = self.d_alpha.data[dest_pid]

        return alpha * (1.0 + beta1 * rab.x + beta2 * rab.y)

    cdef double bonnet_and_lok_gradient_correction(self, size_t dest_pid,
                                                   cPoint * grad):
        """ Correct the gradient of the kernel """

        cdef double x, y, z

        cdef double l11, l12, l13, l21, l22, l23, l31, l32, l33

        l11 = self.bl_l11.data[dest_pid]
        l12 = self.bl_l12.data[dest_pid]
        l13 = self.bl_l13.data[dest_pid]
        l22 = self.bl_l22.data[dest_pid]
        l23 = self.bl_l23.data[dest_pid]
        l33 = self.bl_l33.data[dest_pid]

        l21 = self.bl_l12.data[dest_pid]
        l31 = self.bl_l13.data[dest_pid]
        l32 = self.bl_l23.data[dest_pid]

        x = grad.x; y = grad.y; z = grad.z

        grad.x = l11 * x + l12 * y + l13 * z
        grad.y = l21 * x + l22 * y + l23 * z
        grad.z = l31 * x + l32 * y + l33 * z        

################################################################################
# `CSPHFunctionParticle` class.
################################################################################
cdef class CSPHFunctionParticle(SPHFunctionParticle):
    """ `SPHFunctionParticle` class for use of corrected SPH (CSPH) operations
    
    In this case numerator and denominator are computed for each neighbor
    particle and finally the numerator is divided with the denominator.
    A more efficient way if multiple such funcs are needed may be to do these
    operations in separate funcs so the results can be reused
    """

    cdef void eval_single(self, size_t i, object dst_indices, object nbr_cells,
                          double t=0, double dt=0):
        """ Computes contribution of all neighbors on particle at dest_pid """
        cdef double result[3]
        cdef double dnr[3] # denominator
        cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        cdef size_t nnbrs = nbrs.length
        if self.exclude_self:
            if self.src is self.dest:
                # this works because nbrs has self particle in last position
                nnbrs -= 1
        
        result[0] = result[1] = result[2] = 0.0
        dnr[0] = dnr[1] = dnr[2] = 0.0

        tag = self.dest.get("tag", only_real_particles=False)
        print "Evaluating for particle %d with tag %d"%(dest_pid, tag[dest_pid])
        if tag[dest_pid] == 1:
            print "nnbrs = %d"%(nnbrs)

        # REMOVE THIS!
        dest_pid = 1
        for j in range(nnbrs):
            self.eval_nbr_csph(nbrs.data[j], dest_pid, self.kernel, result, dnr)
        
        for m in range(3):
            if dnr[m] != 0.0:
                result[m] /= dnr[m]
    
    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double * result, double * dnr):
        """ Compute influence when denominator is separately affected by nbrs
        
        This is used in cases such as CSPH where the summation if weighted
        by the kernel sum of all the neighboring particles
        """
        raise NotImplementedError, 'CSPHFunctionParticle.evaleval_nbr_csph()'
