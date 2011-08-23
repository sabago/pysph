"""Functions for Inutsuks's Godunov SPH"""

from euler1d cimport compute_star, sample
from common cimport sqrt, fabs

from pysph.base.particle_array cimport ParticleArray, LocalReal
from pysph.base.kernels cimport KernelBase
from pysph.base.point cimport *     

cdef inline double Cij(double Vi, double Vj, double delsij):
    return (Vi-Vj)/delsij

cdef inline double Dij(double Vi, double Vj, double delsij):
    return 0.5 * (Vi + Vj)

cdef inline double Vij2(double h, double Vi, double Vj, double delsij):
    cdef double cij = Cij(Vi, Vj, delsij)
    cdef double dij = Dij(Vi, Vj, delsij)

    return 0.25 * h * h * cij*cij + dij*dij

cdef inline double sstarij(double h, double Vi, double Vj, double delsij):
    cdef double cij = Cij(Vi, Vj, delsij)
    cdef double dij = Dij(Vi, Vj, delsij)
    cdef double vij2 = Vij2(h, Vi, Vj, delsij)

    return 0.5*h*h*cij*dij/vij2

cdef inline _godunov_solution(int dest_pid, int source_pid,
                              double gamma, double rhol, double rhor,
                              double pl, double pr,
                              double ul, double ur):

    # compute the star values
    cdef double pm, um
    pm, um = compute_star(dest_pid, source_pid, gamma, rhol, rhor, pl, pr, ul, ur)
    
    # get the solution
    cdef double pij, vij, rhoij
    rhoij, vij, pij = sample(pm, um, 0, rhol, rhor, pl, pr, ul, ur, gamma)

    return pij, vij, rhoij

def godunov_solution(dest_pid, source_pid, gamma, rhol, rhor, pl, pr,
                     ul, ur):
    pij, vij, rhoij = _godunov_solution(dest_pid, source_pid, gamma, rhol,
                                        rhor, pl, pr, ul, ur)
    return pij, vij, rhoij

cdef class GSPHMomentumEquation(SPHFunctionParticle):

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, double gamma=1.4, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = "gsph-mom"
        self.tag = "velocity"
        self.gamma = gamma

    def set_src_dst_reads(self):
        pass

    def _set_extra_cl_args(self):
        pass        

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                   KernelBase kernel, double *nr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        cdef cPoint grad, grada, gradb

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef double Va = 1.0/rhoa
        cdef double Vb = 1.0/rhob
        
        cdef cPoint va = cPoint_new(self.d_u.data[dest_pid],
                                    self.d_v.data[dest_pid],
                                    self.d_w.data[dest_pid])

        cdef cPoint vb = cPoint_new(self.s_u.data[source_pid],
                                    self.s_v.data[source_pid],
                                    self.s_w.data[source_pid])
        
        # set up the local origin
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        cdef double delsab = cPoint_length(rab)

        cdef cPoint uvec, origin
        cdef double rhol, rhor, pl, pr, ul, ur
        cdef double pab, rhoab, uab
        cdef double vab2_a, vab2_b
        cdef double _ha, _hb

        if delsab > 1e-15:
            uvec = normalized(rab)
        
            # get the left and right states for the Riemann solver
            rhol = rhob
            rhor = rhoa

            pl = self.s_p.data[source_pid]
            pr = self.d_p.data[dest_pid]

            ul = cPoint_dot(vb, uvec)
            ur = cPoint_dot(va, uvec)
        
            # the Godunov solution at the interface
            pab, uab, rhoab = _godunov_solution(dest_pid, source_pid,
                                                self.gamma, rhol, rhor,
                                                pl, pr, ul, ur)

            vab2_a = Vij2(ha, Va, Vb, delsab)
            vab2_b = Vij2(hb, Va, Vb, delsab)

            _ha = sqrt(2.0)*ha
            _hb = sqrt(2.0)*hb
            
            grada = kernel.gradient(self._dst, self._src, _ha)
            gradb = kernel.gradient(self._dst, self._src, _hb)
            
            # compute the accelerations
            nr[0] += -mb*pab*( (vab2_a*grada.x + vab2_b*gradb.x) )
            nr[1] += -mb*pab*( (vab2_a*grada.y + vab2_b*gradb.y) )
            nr[2] += -mb*pab*( (vab2_a*grada.z + vab2_b*gradb.z) )

cdef class GSPHEnergyEquation(SPHFunctionParticle):

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, double gamma=1.4, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = "gsph-enr"
        self.tag = "energy"
        self.gamma = gamma

    cpdef setup_arrays(self):
        """
        """
        SPHFunctionParticle.setup_arrays(self)
        
        if not self.dest.properties.has_key("ustar"):
            print "%s adding prop ustar"%(self.id)
            self.dest.add_property( {'name':'ustar'} )

        if not self.dest.properties.has_key("vstar"):
            print "%s adding prop vstar"%(self.id)
            self.dest.add_property( {'name':'vstar'} )

        if not self.dest.properties.has_key("wstar"):
            print "%s adding prop wstar"%(self.id)
            self.dest.add_property( {'name':'wstar'} )

        self.d_ustar = self.dest.get_carray("ustar")
        self.d_vstar = self.dest.get_carray("vstar")
        self.d_wstar = self.dest.get_carray("wstar")

    def set_src_dst_reads(self):
        pass

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
        
        cdef cPoint grad, grada, gradb

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef double Va = 1.0/rhoa
        cdef double Vb = 1.0/rhob
        
        cdef cPoint va = cPoint_new(self.d_u.data[dest_pid],
                                    self.d_v.data[dest_pid],
                                    self.d_w.data[dest_pid])

        cdef cPoint vb = cPoint_new(self.s_u.data[source_pid],
                                    self.s_v.data[source_pid],
                                    self.s_w.data[source_pid])

        cdef cPoint vastar = cPoint_new(self.d_ustar.data[dest_pid],
                                        self.d_vstar.data[dest_pid],
                                        self.d_wstar.data[dest_pid])
        
        # set up the local origin
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        cdef double delsab = cPoint_length(rab)

        cdef cPoint uvec, vstarab, v
        cdef double rhol, rhor, pl, pr, ul, ur
        cdef double pab, uab, rhoab
        cdef double vab2_a, vab2_b
        cdef double _ha, _hb

        if delsab > 1e-15:
            uvec = normalized(rab)
        
            # get the left and right states for the Riemann solver
            rhol = rhob
            rhor = rhoa

            pl = self.s_p.data[source_pid]
            pr = self.d_p.data[dest_pid]

            ul = cPoint_dot(vb, uvec)
            ur = cPoint_dot(va, uvec)

            # the Godunov solution at the interface
            pab, uab, rhoab = _godunov_solution(dest_pid, source_pid,
                                                self.gamma, rhol, rhor,
                                                pl, pr, ul, ur)

            # get the projected velocity
            vstarab = cPoint_new(uab*uvec.x,
                                 uab*uvec.y,
                                 uab*uvec.z)

            v = cPoint_sub(vstarab, vastar)
            
            vab2_a = Vij2(ha, Va, Vb, delsab)
            vab2_b = Vij2(hb, Va, Vb, delsab)

            _ha = sqrt(2.0)*ha
            _hb = sqrt(2.0)*hb
            
            grada = kernel.gradient(self._dst, self._src, _ha)
            gradb = kernel.gradient(self._dst, self._src, _hb)

            # evaluate the contribution
            nr[0] -= mb*pab*( vab2_a*cPoint_dot(grada, v) + \
                              vab2_b*cPoint_dot(gradb, v) )
        
cdef class GSPHPositionStepping(SPHFunction):

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True):
        
        SPHFunction.__init__(self, source, dest, setup_arrays)
        
        self.id = 'gsphstep'
        self.tag = "position"

    cpdef setup_arrays(self):

        for prop in ["ustar", "vstar", "wstar"]:
            if not self.dest.properties.has_key(prop):
                msg = "%s Property %s not defined for %"%(self.id, prop,
                                                          self.dest.name)
                raise RuntimeError(msg)
        
        self.d_ustar = self.dest.get_carray("ustar")
        self.d_vstar = self.dest.get_carray("vstar")
        self.d_wstar = self.dest.get_carray("wstar")        

    def set_src_dst_reads(self):
        pass

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        
        for i in range(np):
            if tag_arr.data[i] == LocalReal:
                output1[i] = self.d_ustar.data[i]
                output2[i] = self.d_vstar.data[i]
                output3[i] = self.d_wstar.data[i]
            else:
                output1[i] = output2[i] = output3[i] = 0
