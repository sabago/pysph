
from libc.math cimport sqrt, cos, acos, sin, atan2
cimport cython

cdef extern:
    int isnan(double)

from numpy.linalg import eigh
from numpy import empty
cimport numpy
import numpy

# NOTE: for a symmetric 3x3 matrix M, the notation d,s of denotes the diagonal
# and off-diagonal elements as [c]Points
# d = M[0,0], M[1,1], M[2,2]
# s = M[1,2], M[0,2], M[0,1]

cdef extern from "math.h":
    double fabs(double)

cdef double det(cPoint d, cPoint s):
    ''' determinant of symmetrix matrix '''
    return d.x*d.y*d.z + 2*s.x*s.y*s.z - d.x*s.x*s.x - d.y*s.y*s.y - d.z*s.z*s.z

cpdef double py_det(diag, side):
    ''' determinant of symmetrix matrix '''
    cdef cPoint d
    d.x, d.y, d.z = diag
    cdef cPoint s
    s.x, s.y, s.z = side
    return det(d, s)

cdef cPoint get_eigenvalues(cPoint d, cPoint s):
    ''' eigenvalues of symmetric matrix '''
    cdef double m = (d.x+d.y+d.z)/3
    cdef cPoint Kd = cPoint_sub(d, cPoint_new(m,m,m)) # M-m*eye(3);
    cdef cPoint Ks = s
    
    cdef double q = det(Kd, Ks)/2
    
    cdef double p = 0
    p += Kd.x*Kd.x + 2*Ks.x*Ks.x
    p += Kd.y*Kd.y + 2*Ks.y*Ks.y
    p += Kd.z*Kd.z + 2*Ks.z*Ks.z
    p /= 6.0
    cdef double pi = acos(-1.0)
    cdef double phi = 0.5*pi
    if p != 0 != q:
        phi = acos(q/p**(3.0/2))/3.0
    
    # NOTE: q/p^(3/2) should be in range of [1,-1], but because of numerical errors,
    # it must be checked and in case abs(q) >= abs(p^(3/2)), set phi = 0
    if fabs(q) >= fabs(p**(3.0/2)):
        phi = 0
    
    if phi<0:
        phi += pi/3
    
    cdef cPoint ret
    ret.x = m + 2*sqrt(p)*cos(phi)
    ret.y = m - sqrt(p)*(cos(phi) + sqrt(3)*sin(phi))
    ret.z = m - sqrt(p)*(cos(phi) - sqrt(3)*sin(phi))
    
    return ret

cpdef py_get_eigenvalues(diag, side):
    ''' eigenvalues of symmetric matrix '''
    cdef cPoint d
    d.x, d.y, d.z = diag
    cdef cPoint s
    s.x, s.y, s.z = side
    cdef cPoint ret = get_eigenvalues(d, s)
    return ret.x, ret.y, ret.z


@cython.boundscheck(False)
cdef cPoint get_eigenvector_np(cPoint d, cPoint s, double r):
    ''' eigenvector of symmetric matrix for given eigenvalue `r` using numpy '''
    cdef numpy.ndarray[ndim=2,dtype=numpy.float64_t] mat=empty((3,3)), evec
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] evals
    mat[0,0] = d.x
    mat[1,1] = d.y
    mat[2,2] = d.z
    mat[0,1] = mat[1,0] = s.z
    mat[0,2] = mat[2,0] = s.y
    mat[2,1] = mat[1,2] = s.x
    evals, evec = eigh(mat)
    cdef int idx=0
    cdef double di = fabs(evals[0]-r)
    if fabs(evals[1]-r) < di:
        idx = 1
    if fabs(evals[2]-r) < di:
        idx = 2
    cdef cPoint ret
    ret.x = evec[idx,0]
    ret.y = evec[idx,1]
    ret.z = evec[idx,2]
    return ret

cdef cPoint get_eigenvector(cPoint d, cPoint s, double r):
    ''' get eigenvector of symmetric 3x3 matrix for given eigenvalue `r`

    uses a fast method to get eigenvectors with a fallback to using numpy '''
    cdef cPoint ret
    ret.x = s.z*s.x - s.y*(d.y-r) # a_12 * a_23 - a_13 * (a_22 - r)
    ret.y = s.z*s.y - s.x*(d.x-r) # a_12 * a_13 - a_23 * (a_11 - r)
    ret.z = (d.x-r)*(d.y-r) - s.z*s.z # (a_11 - r) * (a_22 - r) - a_12^2
    cdef double norm = cPoint_length(ret)
    
    if norm *1e14 <= fabs(r):
        # its a zero, let numpy get the answer
        return get_eigenvector_np(d, s, r)
        
    return cPoint_scale(ret, 1.0/norm)

cpdef py_get_eigenvector(diag, side, double r):
    ''' get eigenvector of a symmetric matrix for given eigenvalue `r` '''
    cdef cPoint d
    d.x, d.y, d.z = diag
    cdef cPoint s
    s.x, s.y, s.z = side
    cdef cPoint ret = get_eigenvector(d, s, r)
    return ret.x, ret.y, ret.z

cdef void transform(double A[3][3], double P[3][3], double res[3][3]):
    ''' compute the transformation P.T*A*P and add it into result '''
    for i in range(3):
        for j in range(3):
            #res[i][j] = 0
            for k in range(3):
                for l in range(3):
                    res[i][j] += P[k][i]*A[k][l]*P[l][j] # P.T*A*P

cdef void transform2(cPoint A, double P[3][3], double res[3][3]):
    ''' compute the transformation P.T*A*P and add it into result
    
    A is diagonal '''
    for i in range(3):
        for j in range(3):
            #res[i][j] = 0
            for k in range(3):
                # l = k
                #for l in range(3):
                res[i][j] += P[k][i]*(&A.x)[k]*P[k][j] # P.T*A*P

def py_transform2(A, P):
    cdef double res[3][3]
    cdef double cP[3][3]
    for i in range(3):
        for j in range(3):
            res[i][j] = 0
            cP[i][j] = P[i][j]
    cdef Point cA = Point(*A)
    
    transform2(cA.data, cP, res)
    ret = empty((3,3))
    for i in range(3):
        for j in range(3):
            ret[i][j] = res[i][j]
    return ret

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
        stress = [[self.stress+str(i)+str(j) for j in range(i)] for i in range(3)]
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


cdef class StressRateD(DivVStressFunction):
    ''' evaluate diagonal terms of deviatoric stress rate '''

    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 str stress='sigma', str shear_mod='G', xsph=True, dim=3, *args, **kwargs):
        DivVStressFunction.__init__(self, source, dest,
                                    setup_arrays, stress, *args, **kwargs)
        self.G = shear_mod
        self.xsph = xsph
        self.dim = dim
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
        v = self.d_v.get_npy_array()

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
        for p in range(3): # result
            j = i = p
            res[p] += 2*self.s_G/3.0*(2*gV[i][j])
            for k in range(3): # i==j stress term
                res[p] += self._d_s[i][j][dest_pid]*(gV[i][k]-gV[k][j])

        cdef double tr = (res[0] + res[1] + res[2])/self.dim
        for p in range(self.dim):
            result[p] = res[p] - tr
        for p in range(self.dim, 3):
            result[p] = 0

    
cdef class StressRateS(DivVStressFunction):
    ''' evaluate off-diagonal terms of deviatoric stress rate '''
    
    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 str stress='sigma', str shear_mod='G', xsph=True, *args, **kwargs):
        DivVStressFunction.__init__(self, source, dest,
                                    setup_arrays, stress, *args, **kwargs)
        self.G = shear_mod
        self.xsph = xsph
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

        result[0] = result[1] = result[2] = 0.0
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
            result[p] += self.s_G*(gV[i][j]+gV[j][i])
            for k in range(3):
                result[p] += 0.5*(self._d_s[i][k][dest_pid]*(gV[j][k]-gV[k][j]) + self._d_s[k][j][dest_pid]*(gV[i][k]-gV[k][i]))


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


cdef class MonaghanArtStress(StressFunction):
    """ Computes acceleration from artificial stress """

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, str stress='sigma',
                 double eps=0.3, double n=4, dim=3, **kwargs):

        StressFunction.__init__(self, source, dest, setup_arrays, stress,
                                     **kwargs)

        self.eps = eps
        self.n = n
        self.dim = dim

        self.id = 'monaghan_art_stress'
        self.tag = "velocity"
        
        st = [self.stress+'%d%d'%(j,i) for i in range(3) for j in range(i+1)]
        self.src_reads.extend( ['u','v','w','p','cs'] + st )
        self.src_reads.extend( ['u','v','w','p','cs','rho'] + st )

    def set_src_dst_reads(self):
        StressFunction.set_src_dst_reads(self)
        self.src_reads += ['x','y','z','h','rho','m']
        self.dst_reads += ['x','y','z','h','rho','m']

    cpdef setup_iter_data(self):
        StressFunction.setup_iter_data(self)
        self.rho0 = self.dest.constants['rho0']

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                           KernelBase kernel, double *result):
        if self.dim==2:
            self.eval_nbr_2D(source_pid, dest_pid, kernel, result)
        else:
            self.eval_nbr_gen(source_pid, dest_pid, kernel, result)

    cdef void eval_nbr_2D(self, size_t source_pid, size_t dest_pid,
                          KernelBase kernel, double *result):
        cdef double theta, c, s, sb_xx, sb_yy, s_xx, s_yy, s_xy
        cdef double Rb_xx, Rb_yy, R_xx=0, R_yy=0, R_xy=0
        cdef double mb = self.s_m.data[source_pid]
        cdef double d_rho = self.d_rho.data[dest_pid]
        cdef double s_rho = self.s_rho.data[source_pid]

        # for self point
        s_xx = self._d_s[0][0][dest_pid]
        s_xy = self._d_s[0][1][dest_pid]
        s_yy = self._d_s[1][1][dest_pid]
        theta = atan2(2*s_xy, s_xx-s_yy)
        c = cos(theta)
        s = sin(theta)
        sb_xx = c*c*s_xx + 2*s*c*s_xy+s*s*s_yy
        sb_yy = s*s*s_xx + 2*s*c*s_xy+c*c*s_yy

        if sb_xx > 0:
            Rb_xx = -self.eps*sb_xx/d_rho
        else:
            Rb_xx = 0
        
        if sb_yy > 0:
            Rb_yy = -self.eps*sb_yy/d_rho
        else:
            Rb_yy = 0

        R_xx += c*c*Rb_xx + s*s*Rb_yy
        R_yy += s*s*Rb_xx + c*c*Rb_yy
        R_xy += s*c*(Rb_xx - Rb_yy)

        # for other point
        s_xx = self._s_s[0][0][source_pid]
        s_xy = self._s_s[0][1][source_pid]
        s_yy = self._s_s[1][1][source_pid]
        theta = atan2(2*s_xy, s_xx-s_yy)
        c = cos(theta)
        s = sin(theta)
        sb_xx = c*c*s_xx + 2*s*c*s_xy+s*s*s_yy
        sb_yy = s*s*s_xx + 2*s*c*s_xy+c*c*s_yy

        if sb_xx > 0:
            Rb_xx = -self.eps*sb_xx/s_rho
        else:
            Rb_xx = 0
        
        if sb_yy > 0:
            Rb_yy = -self.eps*sb_yy/s_rho
        else:
            Rb_yy = 0

        R_xx += c*c*Rb_xx + s*s*Rb_yy
        R_yy += s*s*Rb_xx + c*c*Rb_yy
        R_xy += s*c*(Rb_xx - Rb_yy)

        
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
        cdef double f = kernel.function(cPoint_new(h*(self.rho0/d_rho)**(1.0/kernel.dim),0,0),
                                        cPoint_new(0,0,0), h)
        f = kernel.function(self._dst, self._src, h) / f
        
        result[0] += mb * (R_xx*dgrad[0]+R_xy*dgrad[1]) * f**self.n
        result[1] += mb * (R_xy*dgrad[0]+R_yy*dgrad[1]) * f**self.n


    cdef void eval_nbr_gen(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *result):
        cdef int i, j
        cdef double R_a_b[3][3], R[3][3]
        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        
        cdef cPoint sd, ss, Rd=cPoint_new(0,0,0)
        for i in range(3):
            for j in range(3):
                R_a_b[i][j] = 0
        
        # for other point
        symm_to_points(self._s_s, source_pid, sd, ss)

        # compute principal stresses
        cdef cPoint S = get_eigenvalues(sd, ss)
        #print 'principal stress:', dest_pid, ':', S.x, S.y, S.z

        # correction term
        for i in range(3):
            Rd = get_eigenvector(sd, ss, (&S.x)[i])
            for j in range(3):
                R[i][j] = (&Rd.x)[j]
        for i in range(3):
            if (&S.x)[i] > 0:
                (&Rd.x)[i] -= self.eps * (&S.x)[i]*(&S.x)[i]/(rhob*rhob)
        transform2(Rd, R, R_a_b)
        
        # for self point
        Rd = cPoint_new(0,0,0)
        symm_to_points(self._d_s, dest_pid, sd, ss)
        # compute principal stresses
        S = get_eigenvalues(sd, ss)

        # correction term
        for i in range(3):
            Rd = get_eigenvector(sd, ss, (&S.x)[i])
            for j in range(3):
                R[i][j] = (&Rd.x)[j]
        for i in range(3):
            if (&S.x)[i] > 0:
                (&Rd.x)[i] -= self.eps * (&S.x)[i]*(&S.x)[i]/(rhob*rhob)
        transform2(Rd, R, R_a_b)
        
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
        cdef double f = kernel.function(cPoint_new(h*(self.rho0/rhoa)**(1.0/kernel.dim),0,0),
                                        cPoint_new(0,0,0), h)
        f = kernel.function(self._dst, self._src, h) / f
        
        for i in range(3): # result
            for j in range(3): # stress term
                result[i] += mb * R_a_b[i][j] * dgrad[j] * f**self.n


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
        
