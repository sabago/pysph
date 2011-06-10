""" Declarations for the stress SPH functions 

"""

#pysph imports
from pysph.base.point cimport cPoint, cPoint_new, cPoint_sub, cPoint_dot, \
        cPoint_norm, cPoint_scale, cPoint_length, Point

from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase
from pysph.sph.sph_func cimport SPHFunctionParticle, SPHFunction
from pysph.sph.funcs.pressure_funcs cimport MomentumEquation

# d,s are diagonal and off-diagonal elements of a symmetric 3x3 matrix
# d:11,22,33; s:23,13,12
cdef double det(cPoint d, cPoint s)
cpdef double py_det(diag, side)
cdef cPoint get_eigenvalues(cPoint d, cPoint s)
cdef cPoint get_eigenvector(cPoint d, cPoint s, double r)
cpdef py_get_eigenvalues(diag, side)
cpdef py_get_eigenvector(diag, side, double r)
cdef void symm_to_points(double * mat[3][3], long idx, cPoint& d, cPoint& s)
cdef void points_to_symm(mat, idx, d, s)


cdef class StressFunction(SPHFunctionParticle):

    cdef public str stress
    cdef public list d_s
    cdef public list s_s
    cdef double * _d_s[3][3]
    cdef double * _s_s[3][3]

cdef class SimpleStressAcceleration(StressFunction):
    """ SPH function to compute acceleration due to stress """
    pass

cdef class StressAccelerationRL(StressFunction):
    """ Compute stress acceleration with Randles-Libersky renormalization """
    pass

cdef class StressAccelerationCSPM(StressFunction):
    """ Compute stress acceleration with CSPM corrections """
    pass

cdef class StrainEval(StressFunction):
    """ Strain evaluation from velocities """
    pass

cdef class DivVStressFunction(StressFunction):
    cdef void eval_vel_grad(self, size_t dest_pid, double d_u, double d_v,
                            double d_w, double * s_u, double * s_v,
                            double * s_w, double result[3][3], KernelBase kernel,
                            long * nbrs, int nnbrs)

cdef class StressRateD(DivVStressFunction):
    cdef str G
    cdef double s_G
    cdef DoubleArray s_ubar, s_vbar, s_wbar, d_ubar, d_vbar, d_wbar
    cdef bint xsph
    cdef int dim

cdef class StressRateS(DivVStressFunction):
    cdef str G
    cdef double s_G
    cdef DoubleArray s_ubar, s_vbar, s_wbar, d_ubar, d_vbar, d_wbar
    cdef bint xsph

cdef class BulkModulusPEqn(SPHFunction):
    pass

cdef class MonaghanEOS(SPHFunction):
    cdef double gamma

cdef class MonaghanArtStressD(SPHFunction):
    cdef public str stress
    cdef public list d_s
    cdef double * _d_s[3][3]

    cdef double eps

cdef class MonaghanArtStressS(SPHFunction):
    cdef public str stress
    cdef public list d_s
    cdef double * _d_s[3][3]

    cdef double eps

cdef class MonaghanArtStressAcc(SPHFunctionParticle):
    cdef double eps, n, rho0, dr0
    cdef public str R
    cdef public list d_R
    cdef public list s_R
    cdef double * _d_R[3][3]
    cdef double * _s_R[3][3]


cdef class MonaghanArtStress(StressFunction):
    cdef double rho0, eps, n
    cdef int dim
    cdef void eval_nbr_2D(self, size_t source_pid, size_t dest_pid,
                          KernelBase kernel, double *result)
    cdef void eval_nbr_gen(self, size_t source_pid, size_t dest_pid,
                           KernelBase kernel, double *result)
    

cdef class PressureStress(SPHFunction):
    pass

cdef class PressureAcceleration(MomentumEquation):
    pass

cdef class PressureAcceleration2(SPHFunctionParticle):
    pass
