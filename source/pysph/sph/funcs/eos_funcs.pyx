#base imports
from pysph.base.particle_array cimport ParticleArray, LocalReal
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.kernels cimport KernelBase

from pysph.solver.cl_utils import get_real

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

        self.cl_kernel_src_file = "eos_funcs.clt"
        self.cl_kernel_function_name = "IdealGasEquation"
        self.num_outputs = 2

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.dst_reads.extend( ['e','rho'] )

    def _set_extra_cl_args(self):
        self.cl_args.append( get_real(self.gamma, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const gamma' )

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double* result):
        
        cdef double ea = self.d_e.data[dest_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double gamma = self.gamma

        result[0] = (gamma-1.0)*rhoa*ea
        result[1] = sqrt( gamma * result[0]/rhoa )

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.IdealGasEquation(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()

##############################################################################

cdef class TaitEquation(SPHFunction):
    r""" Tait equation of state 
    
    The pressure is set as:

    :math:`$P = B[(\frac{\rho}{\rho_0})^\gamma - 1.0]$`

    where,
    
    :math:`$B = c_0^2 \frac{\rho_0}{\gamma}$`
    
    rho0 -- Reference density (default 1000)
    c0 -- sound speed at the reference density (10 * Vmax)
    Vmax -- estimated maximum velocity in the simulation
    gamma -- usually 7
    

    The sound speed is then set as
    
    :math:`$cs = c_0 * (\frac{\rho}{\rho_0})^((\gamma-1)/2)$`

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

        self.cl_kernel_src_file = "eos_funcs.clt"
        self.cl_kernel_function_name = "TaitEquation"
        self.num_outputs = 2

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.dst_reads.extend( ['rho'] )

    def _set_extra_cl_args(self):
        self.cl_args.append( get_real(self.gamma, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const gamma' )

        self.cl_args.append( get_real(self.co, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const co' )

        self.cl_args.append( get_real(self.ro, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const ro' )

        self.cl_args.append( get_real(self.B, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const B' )

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double* result):

        cdef double gamma = self.gamma

        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double ratio = rhoa/self.ro
        cdef double gamma2 = 0.5*(gamma - 1.0)
        cdef double tmp = pow(ratio, gamma)

        result[0] = (tmp-1.0)*self.B
        result[1] = pow(ratio, gamma2)*self.co

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.IdealGasEquation(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()


cdef class IsothermalEquation(SPHFunction):

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double co = 1.0,
                 double ro = 1000.0, **kwargs):

        SPHFunction.__init__(self, source, dest, setup_arrays,
                             **kwargs)

        self.co = co
        self.ro = ro

        self.id = 'isothermal'
        self.tag = "state"

        self.cl_kernel_src_file = "eos_funcs.clt"
        self.cl_kernel_function_name = "IsothermalEquation"

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.dst_reads.extend( ['rho'] )

    def _set_extra_cl_args(self):
        self.cl_args.append( get_real(self.co, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const co' )

        self.cl_args.append( get_real(self.ro, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const ro' )

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = ['rho']

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double* result):
        
        cdef double rhoa = self.d_rho.data[dest_pid]
        result[0] = self.co*self.co * (rhoa - self.ro)

cdef class MieGruneisenEquation(SPHFunction):

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 double gamma, double ro, double co, double S,
                 bint setup_arrays=True, **kwargs):

        SPHFunction.__init__(self, source, dest, setup_arrays,
                             **kwargs)

        self.gamma = gamma
        self.co = co
        self.ro = ro
        self.S = S

        self.ao = ao = ro * co * co
        self.bo = ao * ( 1 + 2.0*(S - 1.0) )
        self.co = ao * ( 2*(S - 1.0) + 3*(S - 1.0)*(S - 1.0) )

        self.id = 'mie'
        self.tag = "state"

        self.cl_kernel_src_file = "eos_funcs.clt"
        self.cl_kernel_function_name = "MieGruneisenEquation"

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.dst_reads.extend( ['rho'] )

    def _set_extra_cl_args(self):
        pass

    def set_src_dst_reads(self):
        pass

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double* result):
        
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double ea = self.d_e.data[dest_pid]

        cdef double gamma = self.gamma
        cdef double ratio = rhoa/self.ro - 1.0
        cdef double ratio2 = ratio*ratio

        cdef double PH = self.ao * ratio
        if ratio > 0:
            PH = PH + ratio2 * (self.bo + self.co*ratio)

        result[0] = (1.0 - 0.5*self.gamma*ratio)*PH + rhoa*ea*self.gamma
