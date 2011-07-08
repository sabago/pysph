
from libc.math cimport sqrt, cos, acos, sin, atan2
cimport cython

cdef extern:
    int isnan(double)

from numpy import empty
cimport numpy
import numpy

from linalg cimport get_eigenvalvec, transform2, transform2inv

# NOTE: for a symmetric 3x3 matrix M, the notation d,s of denotes the diagonal
# and off-diagonal elements as [c]Points
# d = M[0,0], M[1,1], M[2,2]
# s = M[1,2], M[0,2], M[0,1]

cdef extern from "math.h":
    double fabs(double)


cdef void symm_to_points(double * mat[3][3], long idx, cPoint& d, cPoint& s):
    ''' convert arrays of matrix elements for index idx into d,s components '''
    d.x = mat[0][0][idx]
    d.y = mat[1][1][idx]
    d.z = mat[2][2][idx]
    
    s.x = mat[1][2][idx]
    s.y = mat[0][2][idx]
    s.z = mat[0][1][idx]

cdef void points_to_symm(mat, idx, d, s):
    ''' set values of matrix elements for index idx from d,s components '''
    mat[0][0][idx] = d.x
    mat[1][1][idx] = d.y
    mat[2][2][idx] = d.z
    
    mat[1][2][idx] = s.x
    mat[2][3][idx] = s.y
    mat[1][3][idx] = s.z

cpdef get_K(G, nu):
    ''' Get the bulk modulus from shear modulus and Poisson ratio '''
    return 2.0*G*(1+nu)/(3*(1-2*nu))

cpdef get_nu(G, K):
    ''' Get the Poisson ration from shear modulus and bulk modulus '''
    return (3.0*K-2*G)/(2*(3.0*K+G))

cpdef get_G(K, nu):
    ''' Get the shear modulus from bulk modulus and Poisson ratio '''
    return 3.0*K*(1-2*nu)/(2*(1+nu))



cdef class StressFunction(SPHFunctionParticle):
    ''' base class for functions operating on stress

    this class does setup of stress carrays and implements the conversion
    of stress arrays into data pointer arrays

    '''

    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 str stress='sigma', *args, **kwargs):
        self.stress = stress
        SPHFunctionParticle.__init__(self, source, dest,
                                setup_arrays=setup_arrays)

    def set_src_dst_reads(self):
        stress = []
        for i in range(3):
            for j in range(i+1):
                stress.append(self.stress+str(j)+str(i))
        self.src_reads = stress
        self.dst_reads = stress

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)
        self.d_s = [[None for j in range(3)] for i in range(3)]
        self.s_s = [[None for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(i+1):
                self.d_s[i][j] = self.d_s[j][i] = self.dest.get_carray(
                                                self.stress + str(j) + str(i))
                self.s_s[i][j] = self.s_s[j][i] = self.source.get_carray(
                                                self.stress + str(j) + str(i))
    
    cpdef setup_iter_data(self):
        """Setup data before each iteration"""
        cdef DoubleArray tmp
        for i in range(3):
            for j in range(i+1):
                tmp = self.d_s[j][i]
                self._d_s[i][j] = self._d_s[j][i] = tmp.data
                tmp = self.s_s[j][i]
                self._s_s[i][j] = self._s_s[j][i] = tmp.data
        
    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *result):
        pass

################################################################################
# `SimpleStressAcceleration` class.
################################################################################
cdef class SimpleStressAcceleration(StressFunction):
    """ Computes acceleration from stress """
    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        self.src_reads += ['x','y','z','h','rho','m']
        self.dst_reads += ['x','y','z','h','rho','m']

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *result):
        cdef int i, j
        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint grad = kernel.gradient(self._dst, self._src, h)
        
        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        cdef double * dgrad = &grad.x
        
        for i in range(3): # result
            for j in range(3): # stress term
                add = self.s_m[source_pid] * (self._d_s[i][j][dest_pid]/(rhoa*rhoa) +
                                              self._s_s[i][j][source_pid]/(rhob*rhob)) * dgrad[j]
                result[i] += add


#############################################################################

cdef class StressAccelerationRL(StressFunction):
    """ Compute stress acceleration with Randles-Libersky renormalization """
    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        self.src_reads += ['x','y','z','h','rho','m']
        self.dst_reads += ['x','y','z','h','rho','m']

    # FIXME: implement

cdef class StressAccelerationCSPM(StressFunction):
    """ Compute stress acceleration with CSPM corrections """
    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        self.src_reads += ['x','y','z','h','rho','m']
        self.dst_reads += ['x','y','z','h','rho','m']

    # FIXME: implement
    
cdef class StrainEval(StressFunction):
    """ Compute the strains from velocities """
    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        self.src_reads += ['x','y','z','h','rho','m']
        self.dst_reads += ['x','y','z','h','rho','m']
    
    # FIXME: implement

cdef class DivVStressFunction(StressFunction):
    ''' base class for functions which need to use velocity gradients '''
    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        self.src_reads += ['x','y','z','h','rho','m']
        self.dst_reads += ['x','y','z','h','rho','m']

    cdef void eval_vel_grad(self, size_t dest_pid, double d_u, double d_v,
                            double d_w, double * s_u, double * s_v,
                            double * s_w, double result[3][3], KernelBase kernel,
                            long * nbrs, int nnbrs):
        cdef size_t source_pid
        cdef int i,j,k
        cdef cPoint grad

        #cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef double mb, rhob, h
        cdef double* d_V = [d_u, d_v, d_w]
        cdef double ** s_V = [s_u, s_v, s_w]
        
        # exclude self
        for k in range(nnbrs):
            source_pid = nbrs[k]
            
            mb = self.s_m.data[source_pid]
            #cdef double rhoa = self.d_rho.data[dest_pid]
            rhob = self.s_rho.data[source_pid]
            #cdef double sa = self.d_s.data[dest_pid]
            #cdef double sb = self.s_s.data[source_pid]
            
            h = 0.5*(self.s_h.data[source_pid] + self.d_h.data[dest_pid])
            self._src.x = self.s_x.data[source_pid]
            self._src.y = self.s_y.data[source_pid]
            self._src.z = self.s_z.data[source_pid]
    
            grad = kernel.gradient(self._dst, self._src, h)

            if self.bonnet_and_lok_correction:
                self.bonnet_and_lok_gradient_correction(dest_pid, &grad)
            
            for i in range(3):
                for j in range(3):
                    sub = mb/rhob*(d_V[i]-s_V[i][source_pid])*(&grad.x)[j]
                    result[i][j] -= sub

    def eval_vel_grad_py(self, kernel):
        cdef LongArray nbrs
        cdef double gV[3][3]
        ret = numpy.empty((3,3,self.dest.num_real_particles))
        for dest_pid in range(self.dest.num_real_particles):
            nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
            nnbrs = nbrs.length - 1 # self not needed to find V gradient

            for i in range(3):
                for j in range(3):
                    gV[i][j] = 0

            self.eval_vel_grad(dest_pid, self.d_u.data[dest_pid],
                               self.d_v.data[dest_pid], self.d_w.data[dest_pid],
                               self.s_u.data, self.s_v.data, self.s_w.data,
                               gV, kernel, nbrs.data, nnbrs)
            
            for i in range(3):
                for j in range(3):
                    ret[i,j,dest_pid] = gV[i][j]

        return ret
        


cdef class StressRateD(DivVStressFunction):
    ''' evaluate diagonal terms of deviatoric stress rate '''

    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 str stress='sigma', str shear_mod='G', xsph=True, *args, **kwargs):
        self.G = shear_mod
        self.xsph = xsph
        DivVStressFunction.__init__(self, source, dest,
                                    setup_arrays, stress, *args, **kwargs)
        self.id = 'stress_rate_d'

    cpdef setup_arrays(self):
        StressFunction.setup_arrays(self)
        if self.xsph:
            self.s_ubar = self.source.get_carray('ubar')
            self.s_vbar = self.source.get_carray('vbar')
            self.s_wbar = self.source.get_carray('wbar')
            self.d_ubar = self.dest.get_carray('ubar')
            self.d_vbar = self.dest.get_carray('vbar')
            self.d_wbar = self.dest.get_carray('wbar')

    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        v = ['u', 'v', 'w']
        if self.xsph:
            v += ['ubar', 'vbar', 'wbar']
        self.src_reads += v
        self.dst_reads += v
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        StressFunction.eval(self, kernel, output1, output2, output3)
    
    cpdef setup_iter_data(self):
        """Setup data before each iteration"""
        StressFunction.setup_iter_data(self)
        self.s_G = self.source.constants[self.G]
    
    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        cdef size_t nnbrs = nbrs.length - 1 # self not needed to find V gradient
        cdef double gV[3][3]
        for i in range(3):
            for j in range(3):
                gV[i][j] = 0

        #result[0] = result[1] = result[2] = 0.0

        self.eval_vel_grad(dest_pid, self.d_u.data[dest_pid],
                           self.d_v.data[dest_pid], self.d_w.data[dest_pid],
                           self.s_u.data, self.s_v.data, self.s_w.data,
                           gV, kernel, nbrs.data, nnbrs)
        if self.xsph:
            self.eval_vel_grad(dest_pid, self.d_ubar.data[dest_pid],
                               self.d_vbar.data[dest_pid], self.d_wbar.data[dest_pid],
                               self.s_ubar.data, self.s_vbar.data, self.s_wbar.data,
                               gV, kernel, nbrs.data, nnbrs)

        cdef double * res = [0., 0., 0.]
        cdef double tr = (gV[0][0] + gV[1][1] + gV[2][2])/3.0
        for p in range(3): # result
            j = i = p
            res[p] += 2*self.s_G*(gV[i][j]-tr)
            for k in range(3): # i==j stress term
                res[p] += self._d_s[i][k][dest_pid]*(gV[i][k]-gV[k][j])

        for p in range(3):
            result[p] = res[p]

    
cdef class StressRateS(DivVStressFunction):
    ''' evaluate off-diagonal terms of deviatoric stress rate '''
    
    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 str stress='sigma', str shear_mod='G', xsph=True, *args, **kwargs):
        self.G = shear_mod
        self.xsph = xsph
        DivVStressFunction.__init__(self, source, dest,
                                    setup_arrays, stress, *args, **kwargs)
        self.id = 'stress_rate_s'

    cpdef setup_arrays(self):
        StressFunction.setup_arrays(self)
        if self.xsph:
            self.s_ubar = self.source.get_carray('ubar')
            self.s_vbar = self.source.get_carray('vbar')
            self.s_wbar = self.source.get_carray('wbar')
            self.d_ubar = self.dest.get_carray('ubar')
            self.d_vbar = self.dest.get_carray('vbar')
            self.d_wbar = self.dest.get_carray('wbar')

    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        v = ['u', 'v', 'w']
        if self.xsph:
            v += ['ubar', 'vbar', 'wbar']
        self.src_reads += v
        self.dst_reads += v

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        StressFunction.eval(self, kernel, output1, output2, output3)
    
    cpdef setup_iter_data(self):
        """Setup data before each iteration"""
        StressFunction.setup_iter_data(self)
        self.s_G = self.source.constants[self.G]

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        cdef size_t nnbrs = nbrs.length - 1 # self not needed to find V gradient
        cdef double gV[3][3]
        cdef int i, j
        for i in range(3):
            for j in range(3):
                gV[i][j] = 0

        self.eval_vel_grad(dest_pid, self.d_u.data[dest_pid],
                           self.d_v.data[dest_pid], self.d_w.data[dest_pid],
                           self.s_u.data, self.s_v.data, self.s_w.data,
                           gV, kernel, nbrs.data, nnbrs)
        if self.xsph:
            self.eval_vel_grad(dest_pid, self.d_ubar.data[dest_pid],
                               self.d_vbar.data[dest_pid], self.d_wbar.data[dest_pid],
                               self.s_ubar.data, self.s_vbar.data, self.s_wbar.data,
                               gV, kernel, nbrs.data, nnbrs)

        for p in range(3): # result
            j = 2 - (p==2)
            i = (p==0)
            result[p] = self.s_G*(gV[i][j]+gV[j][i])
            for k in range(3):
                result[p] += 0.5*(self._d_s[i][k][dest_pid]*(gV[j][k]-gV[k][j]) + self._d_s[j][k][dest_pid]*(gV[i][k]-gV[k][i]))


cdef class BulkModulusPEqn(SPHFunction):
    ''' pressure equation P = c_s^2 (rho - rho_0) '''
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 bint setup_arrays=True, *args, **kwargs):
        SPHFunction.__init__(self, source, dest, setup_arrays)
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = ['c_s', 'rho0', 'rho']
        self.dst_reads = ['c_s', 'rho0', 'rho']

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef long i
        cdef double c_s2 = self.dest.constants['c_s']
        c_s2 *= c_s2
        cdef double rho0 = self.dest.constants['rho0']
        for i in range(self.d_rho.length):
            output1.data[i] = c_s2 * (self.d_rho.data[i]-rho0)


cdef class MonaghanEOS(SPHFunction):
    ''' Monaghan's pressure eqn P = c_s^2 ((rho/rho_0)^gamma - 1)/gamma '''
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 bint setup_arrays=True, double gamma=7.0, *args, **kwargs):
        self.gamma = gamma
        SPHFunction.__init__(self, source, dest, setup_arrays)
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','rho','m']
        self.dst_reads = ['x','y','z','h','rho','m']
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef long i
        cdef double c_s2 = self.dest.constants['c_s']
        c_s2 *= c_s2
        cdef double rho0 = self.dest.constants['rho0']
        for i in range(self.d_p.length):
            output1.data[i] = rho0 * c_s2 * ((self.d_rho.data[i]/rho0)**self.gamma-1)/self.gamma

cdef class MonaghanArtStressD(SPHFunction):
    
    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 str stress='sigma', double eps=0.3, *args, **kwargs):
        self.stress = stress
        SPHFunction.__init__(self, source, dest,
                             setup_arrays, stress, *args, **kwargs)
        self.eps = eps
        self.id = 'monaghan_art_stress_d'

    def set_src_dst_reads(self):
        stress = []
        for i in range(3):
            for j in range(i+1):
                stress.append(self.stress+str(j)+str(i))
        self.src_reads = stress + ['rho']
        self.dst_reads = []

    cpdef setup_arrays(self):
        SPHFunction.setup_arrays(self)
        self.d_s = [[None for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(i+1):
                self.d_s[i][j] = self.d_s[j][i] = self.dest.get_carray(
                                                self.stress + str(j) + str(i))

    cpdef setup_iter_data(self):
        """Setup data before each iteration"""
        cdef DoubleArray tmp
        for i in range(3):
            for j in range(i+1):
                tmp = self.d_s[j][i]
                self._d_s[i][j] = self._d_s[j][i] = tmp.data

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        cdef int i, j
        cdef double R_a_b[3][3], R[3][3]
        cdef double rho = self.d_rho[dest_pid]
        
        cdef cPoint sd, ss, Rd
        for i in range(3):
            for j in range(3):
                R_a_b[i][j] = 0
        
        symm_to_points(self._d_s, dest_pid, sd, ss)
        # add the pressure term
        for i in range(3):
            (&sd.x)[i] -= self.d_p[dest_pid]

        # compute principal stresses
        cdef cPoint S = get_eigenvalvec(sd, ss, &R[0][0])
        #print 'stress:', sd, ss
        #print 'principal stress:', dest_pid, ':', S.x, S.y, S.z

        for i in range(3):
            if (&S.x)[i] > 0:
                (&Rd.x)[i] = -self.eps * (&S.x)[i]/(rho*rho)
            else:
                (&Rd.x)[i] = 0.0
        #print 'principal art stress', Rd.x, Rd.y, Rd.z

        transform2inv(Rd, R, R_a_b)

        for p in range(3):
            result[p] = R_a_b[p][p]

cdef class MonaghanArtStressS(SPHFunction):
    
    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 str stress='sigma', double eps=0.3, *args, **kwargs):
        self.stress = stress
        SPHFunction.__init__(self, source, dest,
                             setup_arrays, stress, *args, **kwargs)
        self.eps = eps
        self.id = 'monaghan_art_stress_s'

    def set_src_dst_reads(self):
        stress = []
        for i in range(3):
            for j in range(i+1):
                stress.append(self.stress+str(j)+str(i))
        self.src_reads = stress + ['rho']
        self.dst_reads = []

    cpdef setup_arrays(self):
        SPHFunction.setup_arrays(self)
        self.d_s = [[None for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(i+1):
                self.d_s[i][j] = self.d_s[j][i] = self.dest.get_carray(
                                                self.stress + str(j) + str(i))

    cpdef setup_iter_data(self):
        """Setup data before each iteration"""
        cdef DoubleArray tmp
        for i in range(3):
            for j in range(i+1):
                tmp = self.d_s[j][i]
                self._d_s[i][j] = self._d_s[j][i] = tmp.data

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        cdef int i, j
        cdef double R_a_b[3][3], R[3][3]
        cdef double rho = self.d_rho[dest_pid]
        
        cdef cPoint sd, ss, Rd
        for i in range(3):
            for j in range(3):
                R_a_b[i][j] = 0
        
        symm_to_points(self._d_s, dest_pid, sd, ss)
        # add the pressure term
        for i in range(3):
            (&sd.x)[i] -= self.d_p[dest_pid]

        # compute principal stresses
        cdef cPoint S = get_eigenvalvec(sd, ss, &R[0][0])
        #print 'principal stress:', dest_pid, ':', S.x, S.y, S.z

        for i in range(3):
            if (&S.x)[i] > 0:
                (&Rd.x)[i] = -self.eps * (&S.x)[i]/(rho*rho)
            else:
                (&Rd.x)[i] = 0.0
        #print 'principal art stress', Rd.x, Rd.y, Rd.z

        transform2inv(Rd, R, R_a_b)

        for p in range(3):
            i = (p==0)
            j = 2 - (p==2)
            result[p] = R_a_b[i][j]


cdef class MonaghanArtStressAcc(SPHFunctionParticle):
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, str R='MArtStress',
                 double n=4, **kwargs):

        self.R = R
        self.n = n

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays, **kwargs)

        self.id = 'monaghan_art_stress_acc'
        self.tag = 'velocity'

    def set_src_dst_reads(self):
        R = [self.R+'%d%d'%(j,i) for i in range(3) for j in range(i+1)]
        self.src_reads = R
        self.dst_reads = R
        self.src_reads += ['x','y','z','h','rho','m']
        self.dst_reads += ['x','y','z','h','rho','m']

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)
        self.d_R = [[None for j in range(3)] for i in range(3)]
        self.s_R = [[None for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(i+1):
                self.d_R[i][j] = self.d_R[j][i] = self.dest.get_carray(
                                                      self.R + str(j) + str(i))
                self.s_R[i][j] = self.s_R[j][i] = self.source.get_carray(
                                                      self.R + str(j) + str(i))

    cpdef setup_iter_data(self):
        SPHFunctionParticle.setup_iter_data(self)
        self.rho0 = self.dest.constants['rho0']
        cdef DoubleArray tmp
        for i in range(3):
            for j in range(i+1):
                tmp = self.d_R[j][i]
                self._d_R[i][j] = self._d_R[j][i] = tmp.data
                tmp = self.s_R[j][i]
                self._s_R[i][j] = self._s_R[j][i] = tmp.data
        self.dr0 = self.dest.constants['dr0']
    
    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *result):
        cdef double R_a_b[3][3]
        cdef double s_m = self.s_m.data[source_pid]
        cdef double rho_ab = 0.5*(self.d_rho.data[dest_pid] +
                                  self.s_rho.data[source_pid])
        # cdef double s_rho = self.s_rho.data[source_pid]
        
        for i in range(3):
            for j in range(3):
                R_a_b[i][j] = self._s_R[i][j][source_pid] + self._d_R[i][j][dest_pid]

        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint grad = kernel.gradient(self._dst, self._src, h)
        
        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)
        
        cdef double * dgrad = [grad.x, grad.y, grad.z]
        cdef double f = kernel.function(cPoint_new(self.dr0*(self.rho0/rho_ab)**(1.0/kernel.dim),0,0),
                                        cPoint_new(0,0,0), h)
        f = kernel.function(self._dst, self._src, h) / f
        
        for i in range(3): # result
            for j in range(3): # stress term
                result[i] += s_m * R_a_b[i][j] * dgrad[j] * f**self.n



cdef class PressureStress(SPHFunction):
    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','rho','m']
        self.dst_reads = ['x','y','z','h','rho','m']

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef long i
        for i in range(self.d_p.length):
            output1.data[i] += self.d_p.data[i]
            output2.data[i] += self.d_p.data[i]
            output3.data[i] += self.d_p.data[i]


cdef class PressureAcceleration(MomentumEquation):
    pass

cdef class PressureAcceleration2(SPHFunctionParticle):
    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','rho','m']
        self.dst_reads = ['x','y','z','h','rho','m']

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *result):
        cdef int i
        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        cdef double temp = 0.0

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint grad = kernel.gradient(self._dst, self._src, h)
        
        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        cdef double * dgrad = &grad.x
        
        for i in range(3): # result: i==j
            result[i] += (self.d_p.data[dest_pid]/(rhoa*rhoa) +
                          self.s_p.data[source_pid]/(rhob*rhob)) * dgrad[i]
        
