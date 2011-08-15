from pysph.base.point cimport cPoint_new, cPoint_sub, cPoint_add

from pysph.solver.cl_utils import get_real

cdef extern from "math.h":
    double sqrt(double)
    double fabs(double)
    double pow(double, double)

cdef inline max(double a, double b):
    if a < b:
        return b
    else:
        return a

################################################################################
# `SPHPressureGradient` class.
################################################################################
cdef class SPHPressureGradient(SPHFunctionParticle):
    """
    Computes pressure gradient using the formula 

        INSERTFORMULA

    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True,
                 deltap=None,
                 double n=1,
                 double eps=0.2,
                 **kwargs):
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        if deltap is not None:
            self.deltap = deltap
            if deltap < 0:
                raise RuntimeError("deltap cannot be negative!")
            
            self.with_correction = True

        self.n = n

        self.eps = eps

        self.id = 'pgrad'
        self.tag = "velocity"

        self.cl_kernel_src_file = "pressure_funcs.clt"
        self.cl_kernel_function_name = "SPHPressureGradient"

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','p']
        self.dst_reads = ['x','y','z','h','rho','p','tag']

    def _set_extra_cl_args(self):
        pass

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                   KernelBase kernel, double *nr):
        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double pa = self.d_p.data[dest_pid]
        cdef double pb = self.s_p.data[source_pid]

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double h = 0.5*(ha + hb)

        cdef double temp = 0.0
        cdef double Ra, Rb, rhoa2, rhob2, fab, w, wa, wb

        cdef cPoint grad
        cdef cPoint grada
        cdef cPoint gradb

        cdef cPoint _ra, _rb
        cdef double wdeltap, wdeltap1, wdeltap2
        cdef double artificial_stress = 0.0

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rhoa2 = 1.0/(rhoa*rhoa)
        rhob2 = 1.0/(rhob*rhob)

        temp = -mb * (pa*rhoa2 + pb*rhob2)

        # Artificial pressure
        if self.with_correction:
            if pa < 0:
                Ra = -self.eps * pa
            else:
                Ra = 0.0

            Ra = Ra * rhoa2

            if pb < 0:
                Rb = -self.eps * pb
            else:
                Rb = 0.0

            Rb = Rb * rhob2

            if self.hks:
                wa = kernel.function(self._dst, self._src, ha)
                wb = kernel.function(self._dst, self._src, hb)
                w = 0.5 * (wa + wb)

                _ra = cPoint_new(self.deltap, 0.0, 0.0)
                _rb = cPoint_new(0.0, 0.0, 0.0)
                wdeltap1 = kernel.function(_ra, _rb, ha)

                _ra = cPoint_new(hb * self.deltap, 0.0, 0.0)
                wdeltap2 = kernel.function(_ra, _rb, hb)
                
                wdeltap = 0.5 * (wdeltap1 + wdeltap2)            
            
            else:
                w = kernel.function(self._dst, self._src, h)

                _ra = cPoint(self.deltap, 0.0, 0.0)
                _rb = cPoint(0.0, 0.0, 0.0)
                wdeltap = kernel.function(_ra, _rb, h)
                
            fab = w/wdeltap
            fab = pow(fab, self.n)

            artificial_stress = mb * (Ra + Rb) * fab

        temp = temp + artificial_stress

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

        else:            
            grad = kernel.gradient(self._dst, self._src, h)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)
            
        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.SPHPressureGradient(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()
        

################################################################################
# `MomentumEquation` class.
################################################################################
cdef class MomentumEquation(SPHFunctionParticle):
    """
        INSERTFORMULA

    """
    #Defined in the .pxd file
    #cdef public double alpha
    #cdef public double beta
    #cdef public double gamma
    #cdef public double eta

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, alpha=1, beta=1, gamma=1.4,
                 eta=0.1, deltap=None, double n=1, double eps=0.2,
                 **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
    
        if deltap is not None:
            self.deltap = deltap
            if deltap < 0:
                raise RuntimeError("deltap cannot be negative!")
            
            self.with_correction = True

        self.n = n
        self.eps = eps

        self.id = 'momentumequation'
        self.tag = "velocity"

        self.cl_kernel_src_file = "pressure_funcs.clt"
        self.cl_kernel_function_name = "MomentumEquation"

        self.to_reset = ['dt_fac']

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','p','u','v','w','cs']
        self.dst_reads = ['x','y','z','h','rho','p',
                          'u','v','w','cs','tag']

    cpdef setup_arrays(self):
        """
        """
        SPHFunctionParticle.setup_arrays(self)

        if not self.dest.properties.has_key("dt_fac"):
            self.dest.add_property( {'name':'dt_fac'} )

        self.d_dt_fac = self.dest.get_carray("dt_fac")

        self.dst_reads.append("dt_fac")

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
        cdef double Pa, Pb, rhoa, rhob, rhoab, mb
        cdef double dot, tmp
        cdef double ca, cb, mu, piab, alpha, beta, eta

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef DoubleArray xgc, ygc, zgc

        cdef double hab = 0.5*(ha + hb)
        cdef double dt_fac, rab2

        cdef double rhoa2, rhob2, Ra, Rb, fab, w, wa, wb

        cdef cPoint _ra, _rb
        cdef double wdeltap, wdeltap1, wdeltap2

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        ca = self.d_cs.data[dest_pid]
        cb = self.s_cs.data[source_pid]
        
        #rab = Point_sub(self._dst, self._src)
        cdef cPoint rab, vab
        rab.x = self._dst.x-self._src.x
        rab.y = self._dst.y-self._src.y
        rab.z = self._dst.z-self._src.z
        
        vab.x = self.d_u.data[dest_pid]-self.s_u.data[source_pid]
        vab.y = self.d_v.data[dest_pid]-self.s_v.data[source_pid]
        vab.z = self.d_w.data[dest_pid]-self.s_w.data[source_pid]
        
        dot = cPoint_dot(vab, rab)

        # compute the factor used to determine the viscous time step limit
        rab2 = cPoint_norm(rab)
        eta = self.eta
        dt_fac = fabs( hab * dot / (rab2+eta*eta*hab*hab) )
        self.d_dt_fac.data[dest_pid] = max( self.d_dt_fac.data[dest_pid],
                                            dt_fac )
    
        Pa = self.d_p.data[dest_pid]
        rhoa = self.d_rho.data[dest_pid]        

        Pb = self.s_p.data[source_pid]
        rhob = self.s_rho.data[source_pid]
        mb = self.s_m.data[source_pid]

        rhoa2 = 1.0/(rhoa*rhoa)
        rhob2 = 1.0/(rhob*rhob)

        tmp = Pa*rhoa2 + Pb*rhob2
        
        piab = 0
        if dot < 0:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma

            cab = 0.5 * (ca + cb)

            rhoab = 0.5 * (rhoa + rhob)

            mu = hab*dot
            mu /= (rab2 + eta*eta*hab*hab)
            
            piab = -alpha*cab*mu + beta*mu*mu
            piab /= rhoab
    
        tmp += piab
        tmp *= -mb

        # Artificial pressure
        if self.with_correction:
            Ra = 0.0
            if Pa < 0:
                Ra = -self.eps * Pa

            Ra = Ra * rhoa2

            Rb = 0.0
            if Pb < 0:
                Rb = -self.eps * Pb

            Rb = Rb * rhob2

            if self.hks:
                wa = kernel.function(self._dst, self._src, ha)
                wb = kernel.function(self._dst, self._src, hb)
                w = 0.5 * (wa + wb)

                _ra = cPoint_new(self.deltap, 0.0, 0.0)
                _rb = cPoint_new(0.0,0.0,0.0)

                wdeltap1 = kernel.function(_ra, _rb, ha)

                _ra = cPoint_new(self.deltap, 0.0, 0.0)
                wdeltap2 = kernel.function(_ra, _rb, hb)

                wdeltap = 0.5 * (wdeltap1 + wdeltap2)
            
            else:
                w = kernel.function(self._dst, self._src, hab)

                _ra = cPoint_new(self.deltap, 0.0, 0.0)
                _rb = cPoint_new(0.0, 0.0, 0.0)

                wdeltap = kernel.function(_ra, _rb, hab)

            fab = w/wdeltap
            fab = pow(fab, self.n)
            tmp = tmp + mb*(Ra+Rb)*fab 

        cdef cPoint grad
        cdef cPoint grada
        cdef cPoint gradb

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

        nr[0] += tmp*grad.x
        nr[1] += tmp*grad.y
        nr[2] += tmp*grad.z

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.MomentumEquation(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()

###############################################################################
