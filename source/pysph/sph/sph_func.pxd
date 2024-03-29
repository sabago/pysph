
cimport numpy

# local imports
from pysph.base.particle_array cimport ParticleArray, LocalReal
from pysph.base.carray cimport DoubleArray, IntArray, LongArray, BaseArray
from pysph.base.point cimport Point, cPoint, cPoint_sub
from pysph.base.kernels cimport KernelBase

from pysph.base.kernels cimport KernelBase
from pysph.base.nnps cimport FixedDestNbrParticleLocator

cdef class SPHFunction:
    cdef public ParticleArray source, dest
    cdef public FixedDestNbrParticleLocator nbr_locator
    cdef public num_outputs

    cdef public str name, id
    cdef public str tag
    
    cdef public str h, m, rho, p, e, x, y, z, u, v, w
    cdef public str tmpx, tmpy, tmpz, type
    cdef public str cs

    # a list of destination props that may need resetting at the calc level
    cdef public list to_reset
            
    # Lists on a per-function basis indicating which particle
    # arrays from the source and destination arrays are read.
    
    cdef public list src_reads
    cdef public list dst_reads

    # SPH kernel
    cdef public KernelBase kernel

    # OpenCL Context
    cdef public object context

    # OpenCL kernel source file and function name
    cdef public str cl_kernel_src_file
    cdef public str cl_kernel_function_name

    # OpenCL program and kernel
    cdef public object cl_kernel
    cdef public object cl_program

    # OpenCL kernel launch parameters
    cdef public tuple global_sizes
    cdef public tuple local_sizes

    # OpenCL kernel arguments
    cdef public list cl_args
    cdef public list cl_args_name

    # OpenCL neighbor locator
    cdef public object cl_locator
    
    cdef public DoubleArray s_h, s_m, s_rho
    cdef public DoubleArray s_x, s_y, s_z
    cdef public DoubleArray s_u, s_v, s_w
    cdef public DoubleArray s_p, s_e    
    cdef public DoubleArray s_cs

    cdef public DoubleArray d_h, d_m, d_rho
    cdef public DoubleArray d_x, d_y, d_z
    cdef public DoubleArray d_u, d_v, d_w
    cdef public DoubleArray d_p, d_e    
    cdef public DoubleArray d_cs    

    cpdef setup_arrays(self)
    cpdef setup_iter_data(self)

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3)

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double *result)

################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunctionParticle(SPHFunction):
    cdef cPoint _src, _dst
    cdef bint exclude_self
    
    #rkpm first order correction terms
    cdef public DoubleArray rkpm_beta1, rkpm_beta2, rkpm_beta3
    cdef public DoubleArray rkpm_alpha, rkpm_dalphadx, rkpm_dalphady
    cdef public DoubleArray rkpm_dbeta1dx, rkpm_dbeta1dy
    cdef public DoubleArray rkpm_dbeta2dx, rkpm_dbeta2dy

    cdef public bint bonnet_and_lok_correction
    cdef public bint rkpm_first_order_correction
    #bonnet and lok correction terms ONLY FOR THE DESTINATION!!!
    cdef public DoubleArray bl_l11, bl_l12, bl_l13, bl_l22, bl_l23, bl_l33

    # type of kernel symmetrization to use
    cdef public bint hks    

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double* result)

    cdef double rkpm_first_order_kernel_correction(self, size_t dest_pid)

    cdef double rkpm_first_order_gradient_correction(self, size_t dest_pid)

    cdef double bonnet_and_lok_gradient_correction(self, size_t dest_pid,
                                                   cPoint* grad)

################################################################################
# `CSPHFunctionParticle` class.
################################################################################
cdef class CSPHFunctionParticle(SPHFunctionParticle):
    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double* result, double* dnr)

