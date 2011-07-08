from pysph.base.point cimport cPoint, cPoint_new, cPoint_sub, cPoint_dot, cPoint_norm, Point

from libc.math cimport abs, sqrt, cos, acos, sin


################################################################################
# `DamageEquation` class.
################################################################################
cdef class DamageEquation(StressFunction):
    # FIXME: implement
    def __init__(self, ParticleArray source, dest,  bint setup_arrays=True,
                 alpha=1.0, mfac=6, **kwargs):

        StressFunction.__init__(self, source, dest, setup_arrays,
                                      **kwargs)

        self.alpha = alpha
        self.mfac = mfac
        self.id = 'energyequation'
        self.tag = "energy"

        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = ['u','v','w','p','cs', 'e_t']
        self.dst_reads = ['u','v','w','p','cs', 'e_t']
    
    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        # result = d(D^(1/3))/dt
        cs = self.d_cs[dest_pid]
        R = self.d_h[dest_pid]
        cdef cPoint sd, ss
        symm_to_points(self._d_s, dest_pid, sd, ss)
        e = max(get_eigenvalues(sd, ss).values())/self.dest.E

        if e<self.dest.e_t[dest_pid]:
            result[0] = 0
        else:
            result[0] = ((cs/R)**3 + ((self.mfac+3)/3 *(self.alpha*e)**(1/3.0))*3)*(1/3.0)
        
