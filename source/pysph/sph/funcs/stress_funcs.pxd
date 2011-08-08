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

cdef void symm_to_points(double * mat[3][3], long idx, cPoint& d, cPoint& s)
cdef void points_to_symm(mat, idx, d, s)

cpdef get_K(G, nu)
cpdef get_nu(G, K)
cpdef get_G(K, nu)

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
    cdef double eps, dr0
    cdef double deltap, rho0, n
    
    cdef public str R
    cdef public list d_R
    cdef public list s_R
    cdef double * _d_R[3][3]
    cdef double * _s_R[3][3]

    

cdef class PressureStress(SPHFunction):
    pass

cdef class PressureAcceleration(MomentumEquation):
    pass

cdef class PressureAcceleration2(SPHFunctionParticle):
    pass

cdef class HookesDeviatoricStressRate3D(SPHFunction):

    cdef public double shear_mod

    cdef DoubleArray d_S_00, d_S_01, d_S_02
    cdef DoubleArray d_S_10, d_S_11, d_S_12
    cdef DoubleArray d_S_20, d_S_21, d_S_22

    cdef DoubleArray d_v_00, d_v_01, d_v_02
    cdef DoubleArray d_v_10, d_v_11, d_v_12
    cdef DoubleArray d_v_20, d_v_21, d_v_22

    cpdef tensor_eval(self, KernelBase kernel,
                      DoubleArray output1, DoubleArray output2,
                      DoubleArray output3, DoubleArray output4,
                      DoubleArray output5, DoubleArray output6,
                      DoubleArray output7, DoubleArray output8,
                      DoubleArray output9)

cdef class HookesDeviatoricStressRate2D(SPHFunction):

    cdef public double shear_mod

    cdef DoubleArray d_S_00, d_S_01
    cdef DoubleArray d_S_10, d_S_11

    cdef DoubleArray d_v_00, d_v_01
    cdef DoubleArray d_v_10, d_v_11

    cpdef tensor_eval(self, KernelBase kernel,
                      DoubleArray output1, DoubleArray output2,
                      DoubleArray output3, DoubleArray output4,
                      DoubleArray output5, DoubleArray output6,
                      DoubleArray output7, DoubleArray output8,
                      DoubleArray output9)

cdef class MonaghanArtificialStress(SPHFunction):
    cdef public list d_s
    cdef double *_d_s[3][3]

    cdef public double eps

    cpdef tensor_eval(self, KernelBase kernel)    

cdef class MomentumEquationWithStress2D(SPHFunctionParticle):
    cdef public double deltap
    cdef public double n

    cdef public bint with_correction

    # deviatoric stress components
    cdef DoubleArray d_S_00, d_S_01
    cdef DoubleArray d_S_10, d_S_11

    cdef DoubleArray s_S_00, s_S_01
    cdef DoubleArray s_S_10, s_S_11

    # artificial stress components
    cdef DoubleArray d_R_00, d_R_01
    cdef DoubleArray d_R_10, d_R_11

    cdef DoubleArray s_R_00, s_R_01
    cdef DoubleArray s_R_10, s_R_11
    
cdef class EnergyEquationWithStress2D(SPHFunctionParticle):
    cdef public double deltap
    cdef public double n

    cdef public bint with_correction

    # deviatoric stress components
    cdef DoubleArray d_S_00, d_S_01
    cdef DoubleArray d_S_10, d_S_11

    cdef DoubleArray s_S_00, s_S_01
    cdef DoubleArray s_S_10, s_S_11

    # artificial stress components
    cdef DoubleArray d_R_00, d_R_01
    cdef DoubleArray d_R_10, d_R_11

    cdef DoubleArray s_R_00, s_R_01
    cdef DoubleArray s_R_10, s_R_11


cdef class VonMisesPlasticity2D(SPHFunction):
    cdef public double flow_stress
    cdef public double fac

    # deviatoric stress components
    cdef DoubleArray d_S_00, d_S_01
    cdef DoubleArray d_S_10, d_S_11

