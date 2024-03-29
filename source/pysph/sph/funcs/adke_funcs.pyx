""" File to hold the functions required for the ADKE procedure of Sigalotti """

from libc.math cimport log, exp

from pysph.base.point cimport cPoint, cPoint_new, cPoint_sub, cPoint_dot
from pysph.base.particle_array cimport LocalReal

cdef extern from "math.h":
    double fabs (double)

cdef class SetSmoothingLength(SPHFunction):

    def __init__(self, ParticleArray source, ParticleArray dest,
                 double h0, bint setup_arrays=True):

        SPHFunction.__init__(self, source, dest, setup_arrays)
        
        self.id = 'setsmoothing'
        self.tag = "h"

        self.cl_kernel_src_file = "adke_funcs.cl"
        self.cl_kernel_function_name = "SetSmoothing"
        self.num_outputs = 1

        self.h0 = h0

    def set_src_dst_reads(self):
        pass

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        
        for i in range(np):
            if tag_arr.data[i] == LocalReal:
                output1[i] = self.h0
            else:
                output1[i] = 0    

        # set the destination's dirty bit since new neighbors are needed
        self.dest.set_dirty(True)
        
###############################################################################
# `PilotRho` class.
###############################################################################
cdef class ADKEPilotRho(CSPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, double h0=1.0, **kwargs):
        
        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)
        self.h0 = h0

        self.id = "pilotrho"
        self.tag = "pilotrho"

        self.cl_kernel_src_file = "adke_funcs.cl"
        self.cl_kernel_function_name = "ADKEPilotRho"
        self.num_outputs = 1

    def set_src_dst_reads(self):

        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','m','rho'] )
        self.dst_reads.extend( ['x','y','z','tag'] )

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid, 
                            KernelBase kernel, double *nr, double* dnr):
        """ Compute the contribution from source_pid on dest_pid.

        The expression used is:

        ..math::

        <\rho_p> = \sum_{b=1}^{b=N} m_b\, W(x_a-x_b, h0)

        h0 is a constant that is set upon instance creation.

        """
        cdef double h = self.h0

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double w

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
            
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        w = kernel.function(self._dst, self._src, h)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            dnr[0] += w*mb/rhob

        nr[0] += w*self.s_m.data[source_pid]

###############################################################################
# `ADKESmoothingUpdate` class.
###############################################################################
cdef class ADKESmoothingUpdate(SPHFunction):
    """ Compute the pilot estimate of density for the ADKE algorithm """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, k=1.0, eps=0.0, double h0=1.0,
                 **kwargs):
        
        SPHFunction.__init__(self, source, dest, setup_arrays, **kwargs)

        self.k = k
        self.eps = eps
        self.h0 = h0

        self.id = "adke_smoothing_update"
        self.tag = "h"

        self.cl_kernel_src_file = "adke_funcs.cl"
        self.cl_kernel_function_name = "ADKESmoothingUpdate"
        self.num_outputs = 1

    cpdef setup_arrays(self):
        """
        """
        SPHFunction.setup_arrays(self)

        self.d_rhop = self.dest.get_carray('rhop')

    def set_src_dst_reads(self):

        self.src_reads = []
        self.dst_reads = ['rhop']

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        """ Evaluate the store the results in the output arrays """

        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef DoubleArray rhop = self.d_rhop

        cdef double result, g, log_g
        cdef int i

        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        cdef size_t nr = self.dest.num_real_particles

        log_g = 0.0
        for i in range(np):
            if tag_arr.data[i] == LocalReal:
                log_g += log( rhop.data[i] )

        log_g /= nr
        g = exp(log_g)
        
        for i in range(np):
            if tag_arr.data[i] == LocalReal:
                output1.data[i] = self.h0 * self.k * (g/rhop.data[i])**self.eps

        # set the destination's dirty bit since new neighbors are needed
        self.dest.set_dirty(True)

###############################################################################
# `SPHDivergence` class.
###############################################################################
cdef class SPHVelocityDivergence(SPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 setup_arrays=True, **kwargs):
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = "vdivergence"
        self.tag = "vdivergence"

        self.cl_kernel_src_file = "adke_funcs.cl"
        self.cl_kernel_function_name = "SPHVelocityDivergence"

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','u','v','w','m'] )
        self.dst_reads.extend( ['x','y','z','h','u','v','w','rho','tag'] )

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):
        """ Compute the contribution from source_pid on dest_pid.

        The expression used is:

        ..math::

        <\nabla\,\cdot v>_a = \frac{1}{\rho_a}\sum_{b=1}^{b=N} m_b\,
        (v_b-v_a)\cdot\,\nabla_a\,W_{ab}


        """
        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double h = 0.5 * (ha + hb)

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]

        cdef cPoint grad, grada, gradb, vba

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
            
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        vba = cPoint(self.s_u.data[source_pid] - self.d_u.data[dest_pid],
                     self.s_v.data[source_pid] - self.d_v.data[dest_pid],
                     self.s_w.data[source_pid] - self.d_w.data[dest_pid])

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5
            
            #grad.set((grada.x + gradb.x)*0.5,
            #         (grada.y + gradb.y)*0.5,
            #         (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, h)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        nr[0] += (1.0/rhoa) * mb * cPoint_dot(grad, vba)

###############################################################################
# `ADKEConductionCoeffUpdate` class.
###############################################################################
cdef class ADKEConductionCoeffUpdate(SPHFunction):
    """ Compute the pilot estimate of density for the ADKE algorithm """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, g1=0.0, g2=0.0, **kwargs):
        
        SPHFunction.__init__(self,source,dest,setup_arrays,**kwargs)

        self.g1 = g1
        self.g2 = g2

        self.id = "adke_conduction"
        self.tag = "q"

        self.cl_kernel_src_file = "adke_funcs.cl"

        self.cl_kernel_function_name = "ADKEConductionCoeffUpdate"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','u','v','w','m'] )
        self.dst_reads.extend( ['x','y','z','h','u','v','w','rho','cs','tag'] )

    cpdef setup_arrays(self):
        """
        """
        SPHFunction.setup_arrays(self)

        self.d_div = self.dest.get_carray('div')
        
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        """ Evaluate the store the results in the output arrays Note
        that this function will not work if two or more particle
        arrays contribute as a source. This is because the divergence
        is calculated per source and then the conduction coefficient
        set. We must calculate the divergence from all the
        arrays involved and then perform the array wide operations.

        """

        cdef double g1, g2, ca, ha
        cdef int i

        self.setup_iter_data()
        cdef size_t np = self.dest.num_real_particles

        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef DoubleArray div = self.d_div

        g1 = self.g1
        g2 = self.g2                

        for i in range(np):
            if tag_arr.data[i] == LocalReal:
                ca = self.d_cs.data[i]
                ha = self.d_h.data[i]

                abs_div = fabs( div.data[i] )
                
                # set q_a = g1 h_a c_a + g2 h_a^2 [abs(div_a) - div_a]
            
                output1.data[i] = g1*ca + ( g2 * ha * (abs_div - div.data[i]) )
                output1.data[i] *= ha
