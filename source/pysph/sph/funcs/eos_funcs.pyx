#cython: cdivision=True
#base imports
from pysph.base.particle_array cimport ParticleArray, LocalReal
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.kernels cimport KernelBase

cdef extern from "math.h":
    double pow(double x, double y)
    double sqrt(double x)

cdef class IdealGasEquation(SPHFunction):
    """ Ideal gas equation of state """
    
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double gamma = 1.4, **kwargs):

        SPHFunction.__init__(self, source, dest, setup_arrays)
        self.gamma = gamma

        self.id = 'idealgas'
        self.tag = "state"

        self.dst_reads.extend( ['e','rho', 'p'] )

        self.cl_kernel_src_file = "eos_funcs.cl"

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double* result):
        
        cdef double Pa = self.d_p.data[dest_pid]
        cdef double ea = self.d_e.data[dest_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double gamma = self.gamma

        result[0] = (gamma-1.0)*rhoa*ea
        result[1] = sqrt(ea*(gamma - 1.0))

##############################################################################

cdef class TaitEquation(SPHFunction):
    """ Tait equation of state 
    
    The pressure is set as:

    :math:`P = B[(\frac{\rho}{\rho0})^gamma - 1.0]`

    where,
    
    :math:`B = c0^2 \frac{\rho0}{\gamma}`
    
    rho0 -- Reference density (default 1000)
    c0 -- sound speed at the reference density (10 * Vmax)
    Vmax -- estimated maximum velocity in the simulation
    gamma -- usually 7
    

    The sound speed is then set as
    
    :math:`cs = c0 * (\frac{\rho}{\rho0})^((gamma-1)/2)`

    """
    
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double co = 1.0,
                 double ro = 1000.0, double gamma=7.0, **kwargs):

        SPHFunction.__init__(self, source, dest, setup_arrays,
                             **kwargs)
        self.co = co
        self.ro = ro
        self.gamma = gamma

        self.B = co*co*ro/gamma

        self.id = 'tait'
        self.tag = "state"

        self.dst_reads.append('rho')

        self.cl_kernel_src_file = "eos_funcs.cl"

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double* result):

        cdef double gamma = self.gamma

        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double ratio = rhoa/self.ro
        cdef double gamma2 = 0.5*(gamma - 1.0)
        cdef double tmp = pow(ratio, gamma)

        result[0] = (tmp-1.0)*self.B
        result[1] = pow(ratio, gamma2)*self.co

##############################################################################
