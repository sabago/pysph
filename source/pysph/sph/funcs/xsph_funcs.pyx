from pysph.base.point cimport cPoint, cPoint_new, cPoint_sub, cPoint_dot
from pysph.base.carray cimport DoubleArray

from pysph.solver.cl_utils import get_real

###############################################################################
# `XSPHCorrection' class.
###############################################################################
cdef class XSPHCorrection(CSPHFunctionParticle):
    """ Basic XSPH """

    #Defined in the .pxd file

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double eps = 0.5, **kwargs):

        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)
        self.eps = eps

        self.id = 'xsph'
        self.tag = "position"

        self.cl_kernel_src_file = "xsph_funcs.clt"
        self.cl_kernel_function_name = "XSPHCorrection"

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','u','v','w']
        self.dst_reads = ['x','y','z','h','rho','u','v','w']

    def _set_extra_cl_args(self):

        self.cl_args.append( get_real(self.eps, self.dest.cl_precision) )
        self.cl_args_name.append("REAL const eps")

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):
        """
        The expression used is:

        """
        cdef double temp, w, wa, wb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        cdef double rhoab = 0.5*(self.s_rho.data[source_pid] + \
                                     self.d_rho.data[dest_pid])

        cdef cPoint Va = cPoint(self.d_u.data[dest_pid],
                                self.d_v.data[dest_pid],
                                self.d_w.data[dest_pid])

        cdef cPoint Vb = cPoint(self.s_u.data[source_pid],
                                self.s_v.data[source_pid],
                                self.s_w.data[source_pid])

        cdef cPoint Vba = cPoint_sub(Vb, Va)

        cdef double mb = self.s_m.data[source_pid]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
            
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        if self.hks:
            wa = kernel.function(self._dst, self._src, ha)
            wb = kernel.function(self._dst, self._src, hb)

            w = 0.5 * (wa + wb)

        else:
            w = kernel.function(self._dst, self._src, hab)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            dnr[0] += w*mb/self.s_rho.data[source_pid]

        temp = mb * w/rhoab

        nr[0] += temp*Vba.x*self.eps
        nr[1] += temp*Vba.y*self.eps
        nr[2] += temp*Vba.z*self.eps

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.XSPHCorrection(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()


###############################################################################
# `XSPHDensityRate' class.
###############################################################################
cdef class XSPHDensityRate(SPHFunctionParticle):
    """ Basic XSPHDensityRate """

    #Defined in the .pxd file
    #cdef DoubleArray s_ubar, s_vbar, s_wbar
    
    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, *args, **kwargs):
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays, **kwargs)
        self.num_outputs = 1

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        if self.source is None or self.dest is None:
            return

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        #Setup the XSPH correction terms
        self.s_ubar = self.source.get_carray('ubar')
        self.s_vbar = self.source.get_carray('vbar')
        self.s_wbar = self.source.get_carray('wbar')
        
        self.d_ubar = self.dest.get_carray('ubar')
        self.d_vbar = self.dest.get_carray('vbar')
        self.d_wbar = self.dest.get_carray('wbar')

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','u','v','w', 'ubar', 'vbar', 'wbar']
        self.dst_reads = ['x','y','z','h','rho','u','v','w', 'ubar', 'vbar', 'wbar']

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):
        """
        Perform an SPH interpolation of the property `prop_name`

        The expression used is:

        """
        cdef double h=0.5*(self.s_h.data[source_pid] + \
                               self.d_h.data[dest_pid])

        cdef cPoint Va = cPoint(self.d_u.data[dest_pid]+ \
                                  self.d_ubar.data[dest_pid],

                              self.d_v.data[dest_pid]+ \
                                  self.d_vbar.data[dest_pid],

                              self.d_w.data[dest_pid]+ \
                                  self.d_wbar.data[dest_pid])

        cdef cPoint Vb = cPoint(self.s_u.data[source_pid]+ \
                                  self.s_ubar.data[source_pid],

                              self.s_v.data[source_pid]+ \
                                  self.s_vbar.data[source_pid],
                              
                              self.s_w.data[source_pid]+ \
                                  self.s_wbar.data[source_pid])

        cdef cPoint Vab = cPoint_sub(Va, Vb)
        cdef double mb = self.s_m.data[source_pid]
        cdef double temp

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        #grad = self.kernel_gradient_evaluation[dest_pid][source_pid]
        cdef cPoint grad = kernel.gradient(self._dst, self._src, h)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        temp = cPoint_dot(grad, Vab)
        
        nr[0] += temp*mb

       
###############################################################################
