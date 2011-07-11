"""
Definition file for sph_calc
"""

# numpy import
cimport numpy

# standard imports
from pysph.base.nnps cimport NNPSManager, NbrParticleLocatorBase
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray, LongArray, IntArray
from pysph.base.particle_array cimport ParticleArray
from pysph.sph.sph_func cimport SPHFunctionParticle

from pysph.sph.kernel_correction cimport KernelCorrectionManager

cdef class SPHCalc:
    """ A general purpose class for SPH calculations. """
    cdef public ParticleArray dest

    cdef public int kernel_correction
    cdef public bint bonnet_and_lok_correction
    cdef public bint rkpm_first_order_correction

    cdef public bint nbr_info
    cdef public bint reset_arrays

    cdef public list funcs
    cdef public list nbr_locators
    cdef public list sources

    # a list of destination properties that may need to be reset at every
    # calc invocation. The list is got from the functions.
    cdef public list to_reset

    # properties read of the source and destination arrays
    cdef public list src_reads
    cdef public list dst_reads

    # properties written to the destination arrays
    cdef public dict dst_writes
    cdef public list initial_props

    # flag to call the tensor eval function
    cdef public bint tensor_eval

    # OpenCL context, kernel and command queue
    cdef public object context
    cdef public object queue
    cdef public object cl_kernel

    # The OpenCL kernel source and function name
    cdef public str cl_kernel_src_file
    cdef public str cl_kernel_src
    cdef public str cl_kernel_function_name

    #kernel correction
    cdef public KernelCorrectionManager correction_manager

    cdef public KernelBase kernel    
    cdef public LongArray nbrs 
    cdef public object particles
    cdef public bint integrates
    cdef public list updates, update_arrays
    
    cdef public list from_types, on_types
    cdef public int nupdates
    cdef public int nsrcs
    cdef public str id
    cdef public str tag

    cdef public int dim

    #identifier for the calc's source and destination arrays
    cdef public int dnum
    cdef public str snum

    cdef public NNPSManager nnps_manager

    cpdef sph(self, str output_array1=*, str output_array2=*, 
              str output_array3=*, bint exclude_self=*) 
    
    cpdef sph_array(self, DoubleArray output1, DoubleArray output2,
                    DoubleArray output3, bint exclude_self=*)

    cdef setup_internals(self)
    cpdef check_internals(self)

    cdef reset_output_arrays(self, DoubleArray output1, DoubleArray output2,
                             DoubleArray output3)


    # perform a tensor evaluation. Upto 9 components can be evaluated.
    cpdef tensor_sph(self, str out1=*, str out2=*, str out3=*,
                     str out4=*, str out5=*, str out6=*,
                     str out7=*, str out8=*, str out9=*)

    cpdef tensor_sph_array(self, DoubleArray output1, DoubleArray output2,
                           DoubleArray output3, DoubleArray output4,
                           DoubleArray output5, DoubleArray output6,
                           DoubleArray output7, DoubleArray output8,
                           DoubleArray output9)
