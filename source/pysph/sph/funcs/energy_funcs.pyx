from pysph.base.point cimport cPoint, cPoint_dot, cPoint_new, cPoint_sub,\
     cPoint_norm, normalized

from pysph.solver.cl_utils import get_real

from common cimport compute_signal_velocity, sqrt, fabs

##############################################################################
cdef class EnergyEquationNoVisc(SPHFunctionParticle):
    """ Class to compute interaction of boundary particles on fluid particles
    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)
        self.id = 'energyeqn'
        self.tag = "energy"

        self.cl_kernel_src_file = "energy_funcs.clt"
        self.cl_kernel_function_name = "EnergyEquationNoVisc"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho'] )
        self.dst_reads.extend( ['x','y','z','h','rho','tag'] )

        self.src_reads.extend( ['u','v','w','p'] )
        self.dst_reads.extend( ['u','v','w','p'] )

    def _set_extra_cl_args(self):
        pass
    
    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        """
        cdef double dot, tmp, h
        cdef cPoint vab
        cdef cPoint grad, grada, gradb

        cdef double pa = self.d_p.data[dest_pid]
        cdef double pb = self.s_p.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double mb = self.s_m.data[source_pid]

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        vab.x = self.d_u.data[dest_pid]-self.s_u.data[source_pid]
        vab.y = self.d_v.data[dest_pid]-self.s_v.data[source_pid]
        vab.z = self.d_w.data[dest_pid]-self.s_w.data[source_pid]
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

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

        dot = cPoint_dot(grad, vab)
        tmp = 0.5*mb*(pa/(rhoa*rhoa) + pb/(rhob*rhob))

        nr[0] += tmp*dot

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.EnergyEquationNoVisc(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()
        

##############################################################################
cdef class EnergyEquationAVisc(SPHFunctionParticle):
    """ Class to compute interaction of boundary particles on fluid particles
    """

    #Defined in the .pxd file
    #cdef double beta, alpha, cs, gamma, eta

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True,  beta=1.0, alpha=1.0, 
                 gamma=1.4, eta=0.1, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        self.id = 'energyavisc'
        self.tag = "energy"

        self.cl_kernel_src_file = "energy_funcs.clt"
        self.cl_kernel_function_name = "EnergyEquationAVisc"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho'] )
        self.dst_reads.extend( ['x','y','z','h','rho','tag'] )

        self.src_reads.extend( ['u','v','w','p','cs'] )
        self.dst_reads.extend( ['u','v','w','p','cs'] )
    
    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        """

        cdef double test, gamma, alpha, beta, cs
        cdef double pa, rhoa, pb, rhob, cab, h, mu, prod, rhoab
        cdef cPoint rab, vab
        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        #rab = Point_sub(self._dst, self._src)
        rab.x = self._dst.x-self._src.x
        rab.y = self._dst.y-self._src.y
        rab.z = self._dst.z-self._src.z
        
        #vab = Point_sub(self.tmpva, self.tmpvb)
        vab.x = self.d_u.data[dest_pid]-self.s_u.data[source_pid]
        vab.y = self.d_v.data[dest_pid]-self.s_v.data[source_pid]
        vab.z = self.d_w.data[dest_pid]-self.s_w.data[source_pid]
        
        test = cPoint_dot(vab, rab)
    
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        if test < 0.0:
            gamma = self.gamma 
            alpha = self.alpha
            beta = self.beta
            cs = self.cs
            eta = self.eta

            pa = self.d_p.data[dest_pid]
            pb = self.s_p.data[source_pid]
            rhoa = self.d_rho.data[dest_pid]
            rhob = self.s_rho.data[source_pid]
            mb = self.s_m.data[source_pid]

            cab = 0.5*(self.d_cs.data[dest_pid] + self.s_cs.data[source_pid])

            rhoab = 0.5 * (rhoa + rhob)

            mu = (hab * test) / (cPoint_norm(rab) + eta*eta*hab*hab)

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

            prod  = (-alpha*cab*mu + beta*mu*mu)/(rhoab)
            nr[0] += 0.5 * mb * prod * cPoint_dot(grad, vab)
        else:
            pass
##############################################################################


################################################################################
# `EnergyEquation` class.
################################################################################
cdef class EnergyEquation(SPHFunctionParticle):
    """ Energy Equation
    
    """
    #Defined in the .pxd file
    #cdef public double alpha
    #cdef public double beta
    #cdef public double gamma
    #cdef public double eta

    def __init__(self, ParticleArray source, dest,  bint setup_arrays=True,
                 alpha=1.0, beta=1.0, gamma=1.4, eta=0.1, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        self.id = 'energyequation'
        self.tag = "energy"

        self.cl_kernel_src_file = "energy_funcs.clt"
        self.cl_kernel_function_name = "EnergyEquationWithVisc"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho'] )
        self.dst_reads.extend( ['x','y','z','h','rho','tag'] )

        self.src_reads.extend( ['u','v','w','p','cs'] )
        self.dst_reads.extend( ['u','v','w','p','cs'] )

    def _set_extra_cl_args(self):
        self.cl_args.append( get_real(self.alpha, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const alpha' )

        self.cl_args.append( get_real(self.beta, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const beta' )

        self.cl_args.append( get_real(self.gamma, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const gamma' )

        self.cl_args.append( get_real(self.eta, self.dest.cl_precision) )
        self.cl_args_name.append( 'REAL const eta' )        
        
    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):    
    
        cdef cPoint vab, rab
        cdef double Pa, Pb, rhoa, rhob, rhoab, mb
        cdef double dot, tmp
        cdef double ca, cb, mu, piab, alpha, beta, eta

        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        rab.x = self._dst.x-self._src.x
        rab.y = self._dst.y-self._src.y
        rab.z = self._dst.z-self._src.z
        
        vab.x = self.d_u.data[dest_pid]-self.s_u.data[source_pid]
        vab.y = self.d_v.data[dest_pid]-self.s_v.data[source_pid]
        vab.z = self.d_w.data[dest_pid]-self.s_w.data[source_pid]
        
        dot = cPoint_dot(vab, rab)
    
        Pa = self.d_p.data[dest_pid]
        rhoa = self.d_rho.data[dest_pid]        

        Pb = self.s_p.data[source_pid]
        rhob = self.s_rho.data[source_pid]
        mb = self.s_m.data[source_pid]

        tmp = Pa/(rhoa*rhoa) + Pb/(rhob*rhob)
        
        piab = 0
        if dot < 0:
            alpha = self.alpha
            beta = self.beta
            eta = self.eta
            gamma = self.gamma

            cab = 0.5 * (self.d_cs.data[dest_pid] + self.s_cs.data[source_pid])

            rhoab = 0.5 * (rhoa + rhob)

            mu = hab*dot
            mu /= (cPoint_norm(rab) + eta*eta*hab*hab)

            piab = -alpha*cab*mu + beta*mu*mu
            piab /= rhoab

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

        tmp += piab
        
        tmp = cPoint_dot(grad, vab) * tmp

        nr[0] += 0.5*mb*tmp

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.EnergyEquationWithVisc(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()
        
################################################################################
# `ArtificialHeat` class.
################################################################################
cdef class ArtificialHeat(SPHFunctionParticle):        
    r""" Artificial heat conduction term
    
    Notes
    -----
    The following equation is used in the evaluation
   
    .. raw:: latex

      \[
        \frac{1}{\rho}\nabla \, \cdot(q\nabla(u)) = -\sum_{b=1}^{b=N}
        m_b \frac{(q_a + q_b)(u_a - u_b)}{\rho_{ab}(|\vec{x_a} - \vec{x_b}|^2 +
        (h\eta)^2)} \, (\vec{x_a} - vec{x_b})\cdot \nabla_a W_{ab}
      \]
        
    where :math:`$q_a = h_a (g1 c_a + g2 h_a (abs(div_a) - div_a))$`.
    """
    
    def __init__(self, ParticleArray source, dest,  bint setup_arrays=True,
                 eta=0.1, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays, **kwargs)
        self.eta = eta

        self.id = 'aheat'
        self.tag = "energy"

        self.src_reads.extend( ['u','v','w','p','cs','e','q'] )
        self.dst_reads.extend( ['u','v','w','p','cs','e','q'] )

        self.cl_kernel_src_file = "energy_funcs.clt"
        self.cl_kernel_function_name = "ArtificialHeat"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','e,','u','v','w','cs','q']
        self.dst_reads = ['x','y','z','h','rho','e','u','v','w','cs','q']

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        SPHFunctionParticle.setup_arrays(self)

        self.s_q = self.source.get_carray("q")
        self.d_q = self.dest.get_carray("q")

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
    
    
        cdef cPoint vab, xab
        cdef double rhoa, rhob, rhoab
        cdef double dot, tmp
        cdef double ca, cb, g1, g2, eta
        cdef double ea, eb, diva, divb
        cdef double qa, qb

        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)        
        
        cdef double mb = self.s_m.data[source_pid]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        vab.x = self.d_u.data[dest_pid]-self.s_u.data[source_pid]
        vab.y = self.d_v.data[dest_pid]-self.s_v.data[source_pid]
        vab.z = self.d_w.data[dest_pid]-self.s_w.data[source_pid]
        
        xab = cPoint_sub(self._dst, self._src)
        
        dot = cPoint_dot(xab, vab)

        tmp = 0.0
        if dot < 0:
            eta = self.eta

            ca = self.d_cs.data[dest_pid]
            cb = self.s_cs.data[source_pid]

            rhoab = 0.5 * (self.d_rho.data[dest_pid] + \
                           self.s_rho.data[source_pid])

            mb = self.s_m.data[source_pid]

            eab = self.d_e.data[dest_pid] - self.s_e.data[source_pid]
           
            qa = self.d_q.data[dest_pid]
            qb = self.s_q.data[source_pid]

            tmp = mb * (qa + qb) * eab
            tmp /= (cPoint_norm(xab) + eta*eta*hab*hab)
            
            tmp /= rhoab

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

        nr[0] += tmp * cPoint_dot(grad, xab)

###############################################################################

cdef class EnergyEquationWithSignalBasedViscosity(SPHFunctionParticle):

    def __init__(self, ParticleArray source, ParticleArray dest,
                 double K=1.0, double beta=1.0, double f=1.0, **kwargs):

        self.K = K
        self.beta = beta
        self.f = f
        
        self.id = "energyequationwithsignalbasedviscosity"
        self.tag = "energy"

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays=True,
                                     **kwargs)

        self.cl_kernel_src_file = "energy_funcs.cl"

    def set_src_dst_reads(self):
        pass

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):    
    
        cdef cPoint vab, rab, va, vb
        cdef double Pa, Pb, rhoa, rhob, rhoab, mb
        cdef double dot, tmp
        cdef double ca, cb, mu, piab, alpha, beta, eta
        cdef double ea, eb, eab

        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        rab.x = self._dst.x-self._src.x
        rab.y = self._dst.y-self._src.y
        rab.z = self._dst.z-self._src.z

        va.x = self.d_u.data[dest_pid]
        va.y = self.d_v.data[dest_pid]
        va.z = self.d_w.data[dest_pid]

        vb.x = self.s_u.data[source_pid]
        vb.y = self.s_v.data[source_pid]
        vb.z = self.s_w.data[source_pid]
        
        vab.x = self.d_u.data[dest_pid]-self.s_u.data[source_pid]
        vab.y = self.d_v.data[dest_pid]-self.s_v.data[source_pid]
        vab.z = self.d_w.data[dest_pid]-self.s_w.data[source_pid]
        
        dot = cPoint_dot(vab, rab)
    
        Pa = self.d_p.data[dest_pid]
        rhoa = self.d_rho.data[dest_pid]
        ea = self.d_e.data[dest_pid]
                               
        Pb = self.s_p.data[source_pid]
        rhob = self.s_rho.data[source_pid]
        mb = self.s_m.data[source_pid]
        eb = self.s_e.data[source_pid]

        rhoab = 0.5 * (rhoa + rhob)

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

        temp = Pa/(rhoa*rhoa)
        
        piab = 0
        omegaab = 0
        if dot < 0:
            beta = self.beta
            K = self.K
            f = self.f

            j = normalized(rab)

            vabdotj = cPoint_dot(vab, j)
            vadotj = cPoint_dot(va, j)
            vbdotj = cPoint_dot(vb, j)

            vsig = compute_signal_velocity(self.beta, vabdotj, ca, cb)
            tmp = -K * vsig/rhoab

            piab = tmp * vabdotj

            eab = f*(ea - eb)
            omegaab = tmp * ( 0.5*(vadotj**2 - vbdotj**2) + eab )

        temp = temp * cPoint_dot(grad, vab)
        piab = piab * cPoint_dot(grad, va)
        omegaab = omegaab * cPoint_dot(grad, j)

        nr[0] += mb*temp
        nr[0] += mb*piab
        nr[0] -= mb*omegaab
