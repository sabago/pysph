cdef extern from "math.h":
    double sqrt(double)
    double fabs(double)

cdef inline double max(double a, double b):
    if a < b:
        return b
    else:
        return a

from pysph.base.point cimport cPoint_sub, cPoint_new, cPoint, cPoint_dot, \
        cPoint_norm, cPoint_scale, normalized

from common cimport compute_signal_velocity, compute_signal_velocity2

##############################################################################
# `MonaghanArtificialViscosity` class.
###############################################################################
cdef class MonaghanArtificialViscosity(SPHFunctionParticle):
    """
        INSERTFORMULA

    """
    #Defined in the .pxd file
    #cdef public double c
    #cdef public double alpha
    #cdef public double beta
    #cdef public double gamma

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, alpha=1, beta=1, 
                 gamma=1.4, eta=0.1, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.id = 'momavisc'
        self.tag = "velocity"

        self.cl_kernel_src_file = "viscosity_funcs.cl"
        self.cl_kernel_function_name = "MonaghanArtificialVsicosity"

        self.to_reset = ['dt_fac']

    cpdef setup_arrays(self):
        """
        """
        SPHFunctionParticle.setup_arrays(self)

        if not self.dest.properties.has_key("dt_fac"):
            self.dest.add_property( {'name':'dt_fac'} )

        self.d_dt_fac = self.dest.get_carray("dt_fac")

        self.dst_reads.append("dt_fac")

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','u','v','w','cs']
        self.dst_reads = ['x','y','z','h','p',
                          'u','v','w','cs','rho','tag']

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        cdef cPoint va, vb, vab, rab
        cdef double Pa, Pb, rhoa, rhob, rhoab, mb
        cdef double dot, tmp
        cdef double ca, cb, mu, piab, alpha, beta, eta

        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)
        cdef double rab2, dt_fac

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        va = cPoint(self.d_u.data[dest_pid], self.d_v.data[dest_pid],
                   self.d_w.data[dest_pid])

        vb = cPoint(self.s_u.data[source_pid], self.s_v.data[source_pid],
                   self.s_w.data[source_pid])

        ca = self.d_cs.data[dest_pid]
        cb = self.s_cs.data[source_pid]
        
        rab = cPoint_sub(self._dst, self._src)
        vab = cPoint_sub(va, vb)
        dot = cPoint_dot(vab, rab)

        # compute the factor used to determine the viscous time step limit
        rab2 = cPoint_norm(rab)
        dt_fac = fabs( hab * dot / (rab2 + 0.01*hab*hab) )
        self.d_dt_fac.data[dest_pid] = max( self.d_dt_fac.data[dest_pid],
                                            dt_fac )
    
        rhoa = self.d_rho.data[dest_pid]
        rhob = self.s_rho.data[source_pid]
        mb = self.s_m.data[source_pid]

        piab = 0
        if dot < 0:
            alpha = self.alpha
            beta = self.beta
            eta = self.eta
            gamma = self.gamma

            cab = 0.5 * (ca + cb)

            rhoab = 0.5 * (rhoa + rhob)

            mu = hab*dot
            mu /= (rab2 + eta*eta*hab*hab)
            
            piab = -alpha*cab*mu + beta*mu*mu
            piab /= rhoab
    
        tmp = piab
        tmp *= -mb

        grad = cPoint(0,0,0)
        grada = cPoint(0,0,0)
        gradb = cPoint(0,0,0)

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

            # grad.set((grada.x + gradb.x)*0.5,
            #          (grada.y + gradb.y)*0.5,
            #          (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        nr[0] += tmp*grad.x
        nr[1] += tmp*grad.y
        nr[2] += tmp*grad.z

##############################################################################
# `MomentumEquationSignalBasedViscosity` class.
###############################################################################

cdef class MomentumEquationSignalBasedViscosity(SPHFunctionParticle):

    def __init__(self, ParticleArray source, ParticleArray dest,
                 double K=1.0, double beta=1.0, **kwargs):

        self.K = K
        self.beta = beta
        
        self.id = "momentumequationsignalbasedviscosity"
        self.tag = "velocity"

        if not dest.properties.has_key("dt_fac"):
            msg="Adding prop dt_fac to %s"%(dest.name)
            print msg
            dest.add_property( dict(name="dt_fac") )

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays=True,
                                     **kwargs)

        self.cl_kernel_src_file = "viscosity_funcs.cl"
        self.to_reset = ['dt_fac']

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)

        self.d_dt_fac = self.dest.get_carray("dt_fac")
        self.dst_reads.append("dt_fac")

    def set_src_dst_reads(self):
        pass

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        cdef cPoint grad, grada, gradb
        
        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double rhoab = 0.5*(rhoa + rhob)

        cdef double ca = self.d_cs.data[dest_pid]
        cdef double cb = self.s_cs.data[source_pid]

        cdef cPoint rab, va, vb, vab
        cdef double dot

        cdef double K, beta, vsig, piab, vabdotj
        cdef cPoint j

        cdef double rab2, dt_fac

        va = cPoint_new(self.d_u.data[dest_pid], 
                        self.d_v.data[dest_pid],
                        self.d_w.data[dest_pid])
        
        vb = cPoint_new(self.s_u.data[source_pid],
                        self.s_v.data[source_pid],
                        self.s_w.data[source_pid])
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        vab = cPoint_sub(va,vb)
        rab = cPoint_sub(self._dst,self._src)
        dot = cPoint_dot(vab, rab)

        rab2 = cPoint_norm(rab)
        dt_fac = fabs( hab * dot/ (rab2 + 0.01*hab*hab) )
        self.d_dt_fac.data[dest_pid] = max( self.d_dt_fac.data[dest_pid],
                                            dt_fac )

        piab = 0.0
        if dot < 0:
            K = self.K
            j = normalized(rab)
            vabdotj = cPoint_dot(vab, j)
            
            vsig = compute_signal_velocity(self.beta, vabdotj, ca, cb)
            piab =  -K * vsig * vabdotj/rhoab

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)
            
            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        if self.rkpm_first_order_correction:
            pass
            
        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        nr[0] += -mb*piab*grad.x
        nr[1] += -mb*piab*grad.y
        nr[2] += -mb*piab*grad.z

################################################################################
# `MorrisViscosity` class.
################################################################################
cdef class MorrisViscosity(SPHFunctionParticle):
    """
    Computes pressure gradient using the formula 

        INSERTFORMULA

    """
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str mu='mu', *args, **kwargs):
        """
        Constructor.
        """

        if not ( dest.properties.has_key("mu") ):
            msg = "Dynamic viscosity not defined for %s"%(dest.name)
            raise RuntimeError(msg)

        if not ( source.properties.has_key("mu") ):
            msg = "Dynamic viscosity not defined for %s"%(source.name)
            raise RuntimeError(msg)

        self.mu = mu
        self.id = "morrisvisc"
        self.tag = "velocity"

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays=True,
                                     **kwargs)

        self.cl_kernel_src_file = "viscosity_funcs.cl"
        self.cl_kernel_function_name = "MorrisViscosity"

        self.to_reset = ['dt_fac']

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','u','v','w',self.mu]
        self.dst_reads = ['x','y','z','h','rho','u','v','w','tag',self.mu]

    cpdef setup_arrays(self):
        """
        """
        SPHFunctionParticle.setup_arrays(self)

        self.d_mu = self.dest.get_carray(self.mu)
        self.s_mu = self.source.get_carray(self.mu)

        self.src_reads.append(self.mu)
        self.dst_reads.append(self.mu)

        if not self.dest.properties.has_key("dt_fac"):
            self.dest.add_property( {'name':'dt_fac'} )

        self.d_dt_fac = self.dest.get_carray("dt_fac")

        self.dst_reads.append("dt_fac")

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        cdef cPoint grad, grada, gradb
        
        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double mua = self.d_mu.data[dest_pid]
        cdef double mub = self.s_mu.data[source_pid]

        cdef double temp = 0.0
        cdef cPoint rab, va, vb, vab
        cdef double dot

        cdef double rab2, dt_fac

        va = cPoint_new(self.d_u.data[dest_pid], 
                        self.d_v.data[dest_pid],
                        self.d_w.data[dest_pid])
        
        vb = cPoint_new(self.s_u.data[source_pid],
                        self.s_v.data[source_pid],
                        self.s_w.data[source_pid])
        
        vab = cPoint_sub(va,vb)
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        rab = cPoint_sub(self._dst,self._src)

        rab2 = cPoint_norm(rab)
        dt_fac = fabs( hab * cPoint_dot(vab, rab)/ (rab2) )
        self.d_dt_fac.data[dest_pid] = max( self.d_dt_fac.data[dest_pid],
                                            dt_fac )

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)
            
            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        if self.rkpm_first_order_correction:
            pass
            
        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        dot = cPoint_dot(grad, rab)
            
        temp = mb*(mua + mub)*dot/(rhoa*rhob)
        temp /= (cPoint_norm(rab) + 0.01*hab*hab)

        nr[0] += temp*vab.x
        nr[1] += temp*vab.y
        nr[2] += temp*vab.z
#############################################################################
