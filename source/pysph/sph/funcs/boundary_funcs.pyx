cdef extern from "math.h":
    double fabs(double)

from pysph.base.point cimport cPoint_new, cPoint, cPoint_dot, cPoint_scale

#############################################################################
# `MonaghanBoundaryForce` class.
#############################################################################
cdef class MonaghanBoundaryForce(SPHFunctionParticle):
    """ Class to compute the boundary force for a fluid boundary pair """ 

    #Defined in the .pxd file
    #cdef public double double delp
    #cdef public DoubleArray s_tx, s_ty, s_tz, s_nx, s_ny, s_nz

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=False, double delp = 1.0, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     **kwargs)

        self.id = 'monaghanbforce'
        self.tag = "velocity"
        self.delp = delp

        self.cl_kernel_src_file = "boundary_funcs.cl"
        self.cl_kernel_function_name = "MonaghanBoundaryForce"

    cpdef setup_arrays(self):
        """ Setup the arrays needed for the function """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)
        
        self.s_tx = self.source.get_carray("tx")
        self.s_ty = self.source.get_carray("ty")
        self.s_tz = self.source.get_carray("tz")
        self.s_nx = self.source.get_carray("nx")
        self.s_ny = self.source.get_carray("ny")
        self.s_nz = self.source.get_carray("nz")

        

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho'] )
        self.src_reads.extend( ['tx','ty','tz','nx','ny','nz','cs'] )

        self.dst_reads.extend( ['x','y','z','h','m','cs','tag'] )

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """ Perform the boundary force computation """

        cdef double x, y, nforce, tforce, force
        cdef double beta, q, cs

        cdef double h = self.d_h.data[dest_pid]
        cdef double ma = self.d_m.data[dest_pid]
        cdef double mb = self.s_m.data[source_pid]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        cdef cPoint norm = cPoint(self.s_nx.data[source_pid],
                                  self.s_ny.data[source_pid],
                                  self.s_nz.data[source_pid])
        
        cdef cPoint tang = cPoint(self.s_tx.data[source_pid],
                                  self.s_ty.data[source_pid],
                                  self.s_tz.data[source_pid])

        cs = self.d_cs.data[dest_pid]
        
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        x = cPoint_dot(rab, tang)
        y = cPoint_dot(rab, norm)
        force = 0.0

        q = y/h

        if 0 <= fabs(x) <= self.delp:
            beta = 0.02 * cs * cs/y
            tforce = 1.0 - fabs(x)/self.delp

            if 0 < q <= 2.0/3.0:
                nforce =  2.0/3.0

            elif 2.0/3.0 < q <= 1.0:
                nforce = 2*q*(1.0 - 0.75*q)

            elif 1.0 < q <= 2.0:
                nforce = 0.5 * (2-q)*(2-q)

            else:
                nforce = 0.0
                   
            force = (mb/(ma+mb)) * nforce * tforce * beta

        nr[0] += force*norm.x
        nr[1] += force*norm.y
        nr[2] += force*norm.z

##############################################################################


##############################################################################
cdef class BeckerBoundaryForce(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    #Defined in the .pxd file
    #cdef public double cs

    def __init__(self, ParticleArray source, dest, bint setup_arrays=True,
                 double sound_speed=0.0, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = 'beckerbforce'
        self.tag = "velocity"

        self.sound_speed = sound_speed

        self.cl_kernel_src_file = "boundary_funcs.cl"
        self.cl_kernel_function_name = "BeckerBoundaryForce"

    def set_src_dst_reads(self):
        pass

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        ::math::

            f_ak = \frac{m_k}{m_a + m_k}\tau{x_a,x_k}\frac{x_a -
            x_k}{\lvert x_a - x_k \rvert} where \tau{x_a, x_k} =
            0.02\frac{c_s^2}{\lvert x_a - x_k|\rvert}\begin{cases}
            2/3 & 0 < q < 2/3\\ (2q - 3/2q^2) & 2/3 < q < 1\\
            1/2(2-q)^2 & 1 < q < 2\\ 0 & \text{otherwise}
            \end{cases}
        
        """
        cdef double norm, nforce, force
        cdef double beta, q
        cdef cPoint rab, rabn

        cdef double h = self.d_h.data[dest_pid]
        cdef double ma = self.d_m.data[dest_pid]
        cdef double mb = self.s_m.data[source_pid]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        rab = cPoint_sub(self._dst, self._src)
        norm = cPoint_length(rab)
        rabn = cPoint_scale(rab, 1/norm)

        #Evaluate the normal force
        q = norm/h
        beta = 0.02 * self.sound_speed * self.sound_speed/norm
        
        if q > 0.0 and q <= 2.0/3.0:
            nforce = 2.0/3.0

        elif q > 2.0/3.0 and q <= 1.0:
            nforce = (2.0*q - (3.0/2.0)*q*q)

        elif q > 1.0 and q < 2.0:
            nforce = (0.5)*(2.0 - q)*(2.0 - q)

        force = (mb/(ma+mb)) * nforce * beta
            
        nr[0] += force*rabn.x
        nr[1] += force*rabn.y
        nr[2] += force*rabn.z
##############################################################################

##############################################################################
cdef class LennardJonesForce(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    #Defined in the .pxd file
    #cdef public double D
    #cdef public double ro
    #cdef public double p1, p2

    def __init__(self, ParticleArray source, dest, bint setup_arrays=True,
                 double D=0, double ro=0, double p1=0, double p2=0,
                 **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = 'lenardbforce'
        self.tag = "velocity"
        
        self.D = D
        self.ro = ro
        self.p1 = p1
        self.p2 = p2

        self.cl_kernel_src_file = "boundary_funcs.cl"
        self.cl_kernel_function_name = "LennardJonesForce"

    def set_src_dst_reads(self):
        pass

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        ::math::

        """
        cdef double norm, force, tmp, tmp1, tmp2
        cdef cPoint rab, rabn

        cdef double ro = self.ro
        cdef double D = self.D

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = cPoint_sub(self._dst, self._src)
        norm = cPoint_length(rab)
        rabn = cPoint_scale(rab, 1/norm)

        #Evaluate the normal force
        if norm <= ro:
            tmp = ro/norm
            tmp1 = tmp**self.p1
            tmp2 = tmp**self.p2
            
            force = D*(tmp1 - tmp2)/(norm*norm)
        else:
            force = 0.0
            
        nr[0] += force*rabn.x
        nr[1] += force*rabn.y
        nr[2] += force*rabn.z

##############################################################################
