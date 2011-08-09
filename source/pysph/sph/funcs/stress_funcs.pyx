
from libc.math cimport sqrt, cos, acos, sin, atan2
cimport cython

cdef extern:
    int isnan(double)

from numpy import empty
cimport numpy
import numpy

from linalg cimport get_eigenvalvec, transform2, transform2inv

from pysph.base.particle_array cimport LocalReal

# NOTE: for a symmetric 3x3 matrix M, the notation d,s of denotes the diagonal
# and off-diagonal elements as [c]Points
# d = M[0,0], M[1,1], M[2,2]
# s = M[1,2], M[0,2], M[0,1]


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

cpdef double get_principal_stress_angle2D(double s_00, double s_01,
                                          double s_11, double fac):
    r""" Return the angle in the plane for which shear stresses are zero.

    .. math::

    \tan2\theta_a = \frac{2\sigma^{xy}}{\sigma_a^{xx} - \sigma_a^{yy}}

    Parameters:
    -----------

    s_00 : double
        Plane stress in the first coordinate

    s_01 : double
        Shear stress

    s_11 : dobule
        Plae stress in the second coordinate

    """

    cdef double denominator = s_00 - s_11 + fac
    cdef double numerator = 2.0*s_01
    return 0.5 * atan2(numerator,denominator)

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
    r''' Pressure equation :math:`$P = c_s^2 (\rho - \rho_0)$` '''
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
    r''' s pressure eqn :math:`$P = c_s^2 ((\rho/\rho_0)^\gamma -
    1)/\gamma$` '''
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

cdef class MonaghanArtificialStress(SPHFunction):
    """ Evaluate the artificial stress factors for the momentum equation. """

    def __init__(self, ParticleArray source, ParticleArray dest, setup_arrays=True,
                 double eps=0.3, *args, **kwargs):

        self.eps = eps
        self.id = "monaghanartstress"

        # ensure that the deviatoric stress components are defined
        stress_props = ["S_00", "S_01", "S_02",
                                "S_11", "S_12",
                                        "S_22"]

        dest_props = dest.properties.keys()

        for prop in stress_props:
            if not prop in dest_props:
                msg = "%s adding property %s to array `%s`"%(self.id,
                                                             prop, dest.name)
                print msg
                dest.add_property( dict(name=prop) )

        # add the artificial stress variables
        art_stress = ["R_00", "R_01", "R_02",
                              "R_11", "R_12",
                                      "R_22"]

        for prop in art_stress:
            if not prop in dest_props:
                msg = "%s adding property %s to array `%s`"%(self.id,
                                                             prop, dest.name)
                print msg
                dest.add_property( dict(name=prop) )

        SPHFunction.__init__(self, source, dest, setup_arrays, *args, **kwargs)

    cpdef setup_arrays(self):
        SPHFunction.setup_arrays(self)

        # setup the deviatoric stress components
        self.d_s = [[None for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(i+1):
                self.d_s[i][j] = self.d_s[j][i] = self.dest.get_carray(
                                                "S_" + str(j) + str(i))

    def set_src_dst_reads(self):
        stress = []
        for i in range(3):
            for j in range(3):
                stress.append('S_' + str(j) + str(i))

        self.src_reads = stress + ['rho']
        self.dst_reads = []

    cpdef setup_iter_data(self):
        """Setup data before each iteration"""
        cdef DoubleArray tmp
        for i in range(3):
            for j in range(i+1):
                tmp = self.d_s[j][i]
                self._d_s[i][j] = self._d_s[j][i] = tmp.data

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        """ Overide the basic function to call the tensor eval function """
        self.tensor_eval(kernel)

    cpdef tensor_eval(self, KernelBase kernel):

        # get the tag array 
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        # perform any iteration specific setup
        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        cdef size_t a

        cdef double result[6]
        
        cdef DoubleArray r_00 = self.dest.get_carray("R_00")
        cdef DoubleArray r_01 = self.dest.get_carray("R_01")
        cdef DoubleArray r_02 = self.dest.get_carray("R_02")

        cdef DoubleArray r_11 = self.dest.get_carray("R_11")
        cdef DoubleArray r_12 = self.dest.get_carray("R_12")
        cdef DoubleArray r_22 = self.dest.get_carray("R_22")

        result[0] = result[1] = result[2] = 0.0
        result[1] = result[2] = result[3] = 0.0

        # setup the iteration data
        self.setup_iter_data()

        for a in range(np):
            if tag_arr.data[a] == LocalReal:
                self.eval_single(a, kernel, result)

                r_00.data[a] = result[0]
                r_11.data[a] = result[1]
                r_22.data[a] = result[2]

                r_12.data[a] = result[3]
                r_02.data[a] = result[4]
                r_01.data[a] = result[5]

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):

        cdef int i, j, p
        cdef double rho = self.d_rho[dest_pid]
        cdef double rho2 = 1.0/(rho*rho)

        # the diagonal terms for the stress tensor
        cdef cPoint sd

        # the off diagonal terms for the stress tensor
        cdef cPoint ss

        # the artificial stress terms in the principal directions
        cdef cPoint Rd

        # the Matrix of eigenvectors 
        cdef double R[3][3]

        # the artificial stresses in the original co-ordinates
        cdef double R_a_b[3][3]
        
        # initialize the artificial stress terms to 0
        for i in range(3):
            for j in range(3):
                R_a_b[i][j] = 0

        # get the diagonal/off-diagonal terms from the tensor for this particle
        symm_to_points(self._d_s, dest_pid, sd, ss)

        # add the pressure term to the diagonal terms
        for i in range(3):
            (&sd.x)[i] -= self.d_p[dest_pid]

        # compute principal stresses
        cdef cPoint S = get_eigenvalvec(sd, ss, &R[0][0])

        # perform the correction
        for i in range(3):
            if (&S.x)[i] > 0:
                (&Rd.x)[i] = -self.eps * (&S.x)[i] * rho2
            else:
                (&Rd.x)[i] = 0.0

        # get the artificial stresses in the original frame
        transform2inv(Rd, R, R_a_b)

        # store the result.
        # result[0] = R_00, result[1] = R_11, result[2] = R_22
        # result[3] = R_12, result[4] = R_02, result[5] = R_01 

        for p in range(3):
            result[p] = R_a_b[p][p]

        for p in range(3):
            i = (p==0)
            j = 2 - (p==2)
            result[3+p] = R_a_b[i][j]
        
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
                 double n=4, deltap=1, rho0=1.0,**kwargs):

        self.R = R
        self.n = n

        self.rho0 = rho0
        self.deltap = deltap

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
        
        for i in range(3):
            for j in range(3):
                R_a_b[i][j] = self._s_R[i][j][source_pid] + \
                              self._d_R[i][j][dest_pid]

        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint grad = kernel.gradient(self._dst, self._src, h)

        cdef double ratio = self.rho0/rho_ab
        ratio = pow(ratio, 1.0/kernel.dim)

        cdef cPoint _ra = cPoint_new(self.dr0*ratio, 0.0, 0.0)
        cdef cPoint _rb = cPoint_new(0.0, 0.0, 0.0)
        
        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        cdef double * dgrad = [grad.x, grad.y, grad.z]
        cdef double f = kernel.function(_ra, _rb, h)
        #cdef double f = kernel.function(cPoint_new(self.dr0*(self.rho0/rho_ab)**(1.0/kernel.dim),0,0),
         #                               cPoint_new(0,0,0), h)

        f = kernel.function(self._dst, self._src, h) / f
        #print dest_pid, f**self.n
        
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
        

#############################################################################
# `HookesDeviatoricStressRate` class
#############################################################################
cdef class HookesDeviatoricStressRate3D(SPHFunction):
    r""" Compute the RHS for the rate of change of stress equation (3D)

    .. math::

    \frac{dS^{ij}}{dt} = 2\mu\left(\eps^{ij} -
    \frac{1}{3}\delta^{ij}\eps^{ij}\right) + S^{ik}\Omega^{jk} +
    \Omega^{ik}S^{kj}

    where,

    \eps^{ij} = \frac{1}{2}\left( \frac{\partialv^i}{\partialx^j} +
                                  \frac{\partialv^j}{\partialx^i} \right )

    and

    \Omega^{ij} = \frac{1}{2}\left( \frac{\partialv^i}{\partialx^j} -
                                  \frac{\partialv^j}{\partialx^i} \right )

    """
    
    def __init__(self, ParticleArray source, ParticleArray dest,
                 shear_mod=1.0, bint setup_arrays=True, *args, **kwargs):

        self.id = "stress_rate"
        self.tag = "stress"

        self.shear_mod = shear_mod

        stress_props = ["S_00", "S_01", "S_02",
                        "S_10", "S_11", "S_12",
                        "S_20", "S_21", "S_22"]

        dest_props = dest.properties.keys()

        for prop in stress_props:
            if not prop in dest_props:
                dest.add_property( dict(name=prop) )

        # the velocity gradient tensor props should be defined on the
        # destination. This means, a previous operation of
        # VelocityGradient3D should have been defined for the dest.
        vgrad_props = ["v_00", "v_01", "v_02",
                        "v_10", "v_11", "v_12",
                        "v_20", "v_21", "v_22"]

        for prop in vgrad_props:
            if not prop in dest_props:
                err_msg = """Velocity gradient tensor prop %s not defined.
                This could mean that the required function VelocityGradient3D
                is not defined!"""%(prop)

                raise RuntimeError(err_msg)        

        SPHFunction.__init__(self, source, dest, setup_arrays)

    def set_src_dst_reads(self):
        pass

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)

        # setup the stress arrays
        self.d_S_00 = self.dest.get_carray("S_00")
        self.d_S_01 = self.dest.get_carray("S_01")
        self.d_S_02 = self.dest.get_carray("S_02")

        self.d_S_10 = self.dest.get_carray("S_10")
        self.d_S_11 = self.dest.get_carray("S_11")
        self.d_S_12 = self.dest.get_carray("S_12")

        self.d_S_20 = self.dest.get_carray("S_20")
        self.d_S_21 = self.dest.get_carray("S_21")
        self.d_S_22 = self.dest.get_carray("S_22")

        # setup the velocity gradient tensor arrays
        self.d_v_00 = self.dest.get_carray("v_00")
        self.d_v_01 = self.dest.get_carray("v_01")
        self.d_v_02 = self.dest.get_carray("v_02")

        self.d_v_10 = self.dest.get_carray("v_10")
        self.d_v_11 = self.dest.get_carray("v_11")
        self.d_v_12 = self.dest.get_carray("v_12")

        self.d_v_20 = self.dest.get_carray("v_20")
        self.d_v_21 = self.dest.get_carray("v_21")
        self.d_v_22 = self.dest.get_carray("v_22")

    def setup_calc_updates(self, calc):
        r""" Setup the updates for the stress variable.

        The stress tensor is written as:

        .. math::

        \sigma^{ij} = -P\delta^{ij} + S^{ij}

        S is the deviatoric component of the stress tensor. It's nine
        components are devined as:

        S_00, S_01, S_02
        S_10, S_11, S_12
        S_20, S_21, S_22

        This deviatoric stress component will have to be integrated
        and the corresponding integrator variables are:

        Initial props:
        S_000, S_010, S_020
        S_100, S_110, S_120
        S_200, S_210, S_220

        Step props:
        S_00_a_1, S_01_a_1, S_02_a_1
        S_10_a_1, S_11_a_1, S_12_a_1
        S_20_a_1, S_21_a_1, S_22_a_1

        """
        calc.tensor_eval = True

        updates = []
        for i in range(3):
            for j in range(3):
                updates.append( "S_" + str(i) + str(j) )

        return updates
        
    cpdef tensor_eval(self, KernelBase kernel,
                      DoubleArray output1, DoubleArray output2,
                      DoubleArray output3, DoubleArray output4,
                      DoubleArray output5, DoubleArray output6,
                      DoubleArray output7, DoubleArray output8,
                      DoubleArray output9):

        """ Evaluate the nine components the deviatoric stress rate equation """ 
        
        # get the tag array pointer
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        # perform any iteration specific setup
        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        cdef size_t a

        cdef double result[9]

        for a in range(np):
            if tag_arr.data[a] == LocalReal:
                self.eval_single(a, kernel, result)

                output1.data[a] += result[0]
                output2.data[a] += result[1]
                output3.data[a] += result[2]

                output4.data[a] += result[3]
                output5.data[a] += result[4]
                output6.data[a] += result[5]

                output7.data[a] += result[6]
                output8.data[a] += result[7]
                output9.data[a] += result[8]

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):

        result[0] = result[1] = result[2] = 0.0
        result[3] = result[4] = result[5] = 0.0
        result[6] = result[7] = result[8] = 0.0

        cdef double v_00 = self.d_v_00.data[dest_pid]
        cdef double v_01 = self.d_v_01.data[dest_pid]
        cdef double v_02 = self.d_v_02.data[dest_pid]

        cdef double v_10 = self.d_v_10.data[dest_pid]
        cdef double v_11 = self.d_v_11.data[dest_pid]
        cdef double v_12 = self.d_v_12.data[dest_pid]

        cdef double v_20 = self.d_v_20.data[dest_pid]
        cdef double v_21 = self.d_v_21.data[dest_pid]
        cdef double v_22 = self.d_v_22.data[dest_pid]

        cdef double s_00 = self.d_S_00.data[dest_pid]
        cdef double s_01 = self.d_S_01.data[dest_pid]
        cdef double s_02 = self.d_S_02.data[dest_pid]

        cdef double s_10 = self.d_S_10.data[dest_pid]
        cdef double s_11 = self.d_S_11.data[dest_pid]
        cdef double s_12 = self.d_S_12.data[dest_pid]

        cdef double s_20 = self.d_S_20.data[dest_pid]
        cdef double s_21 = self.d_S_21.data[dest_pid]
        cdef double s_22 = self.d_S_02.data[dest_pid]

        # strain rate tensor is symmetric
        cdef double eps_00 = v_00
        cdef double eps_01 = 0.5 * (v_01 + v_10)
        cdef double eps_02 = 0.5 * (v_02 + v_20)

        cdef double eps_10 = eps_01
        cdef double eps_20 = eps_02

        cdef double eps_11 = v_11
        cdef double eps_12 = 0.5 * (v_12 + v_21)

        cdef double eps_21 = eps_12
        
        cdef double eps_22 = v_22

        # rotation tensor is asymmetric
        cdef double omega_01 = 0.5 * (v_01 - v_10)
        cdef double omega_02 = 0.5 * (v_02 - v_20)

        cdef double omega_10 = -omega_01
        cdef double omega_20 = -omega_02

        cdef double omega_12 = 0.5 * (v_12 - v_21)
        cdef double omega_21 = -omega_12

        cdef double tmp = 2.0*self.shear_mod
        cdef double trace = 1.0/3.0 * (eps_00 + eps_11 + eps_22)

        # S_00
        result[0] = tmp*( eps_00 - trace ) + \
                    ( s_01*omega_01 + s_02*omega_02 ) + \
                    ( s_10*omega_01 + s_20*omega_02 )

        # S_01
        result[1] = tmp*(eps_01) + \
                    ( s_00*omega_10 + s_02*omega_12 ) + \
                    ( s_11*omega_01 + s_21*omega_02 )

        # S_02
        result[2] = tmp*(eps_02) + \
                    ( s_00*omega_20 + s_01*omega_21 ) + \
                    ( s_12*omega_01 + s_22*omega_02 )

        # S_10
        result[3] = tmp*(eps_10) + \
                    ( s_11*omega_01 + s_12*omega_02 ) + \
                    ( s_00*omega_10 + s_20*omega_12 )

        # S_11
        result[4] = tmp*( eps_11 - trace ) + \
                    ( s_10*omega_10 + s_12*omega_12 ) + \
                    ( s_01*omega_10 + s_21*omega_12 )

        # S_12
        result[5] = tmp*eps_12 + \
                    ( s_10*omega_20 + s_11*omega_21 ) + \
                    ( s_02*omega_10 + s_22*omega_12 )

        # S_20
        result[6] = tmp*eps_20 + \
                    ( s_21*omega_01 + s_22*omega_02 ) + \
                    ( s_00*omega_20 + s_10*omega_21 )

        # S_21
        result[7] = tmp*eps_21 + \
                    ( s_20*omega_10 + s_22*omega_12 ) + \
                    ( s_01*omega_20 + s_11*omega_21 )

        # S_22
        result[8] = tmp*( eps_22 - trace)  + \
                    ( s_20*omega_20 + s_21*omega_21 ) + \
                    ( s_02*omega_20 + s_12*omega_21 )


#############################################################################
# `HookesDeviatoricStressRate2D` class
#############################################################################
cdef class HookesDeviatoricStressRate2D(SPHFunction):
    r""" Compute the RHS for the rate of change of stress equation (2D)

    .. raw:: latex

      \[
      \frac{dS^{ij}}{dt} = 2\mu\left(\eps^{ij} -
      \frac{1}{3}\delta^{ij}\eps^{ij}\right) + S^{ik}\Omega^{jk} +
      \Omega^{ik}S^{kj}
      \]

    where,

    :math:`$\eps^{ij} = \frac{1}{2}\left( \frac{\partialv^i}{\partialx^j} +
                                  \frac{\partialv^j}{\partialx^i} \right
                                  )$`

    and

    :math:`$\Omega^{ij} = \frac{1}{2}\left( \frac{\partialv^i}{\partialx^j} -
                                  \frac{\partialv^j}{\partialx^i} \right
                                  )$`

    """
    
    def __init__(self, ParticleArray source, ParticleArray dest,
                 shear_mod=1.0, bint setup_arrays=True, *args, **kwargs):

        self.id = "stress_rate"
        self.tag = "stress"

        self.shear_mod=shear_mod

        stress_props = ["S_00", "S_01",
                        "S_10", "S_11"]

        dest_props = dest.properties.keys()

        for prop in stress_props:
            if not prop in dest_props:
                dest.add_property( dict(name=prop) )

        # the velocity gradient tensor props should be defined on the
        # destination. This means, a previous operation of
        # VelocityGradient2D should have been defined for the dest.
        vgrad_props = ["v_00", "v_01",
                        "v_10", "v_11"]

        for prop in vgrad_props:
            if not prop in dest_props:
                err_msg = """Velocity gradient tensor prop %s not defined.
                This could mean that the required function VelocityGradient2D
                is not defined!"""%(prop)

                raise RuntimeError(err_msg)        

        SPHFunction.__init__(self, source, dest, setup_arrays)

    def set_src_dst_reads(self):
        pass

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)

        # setup the stress arrays
        self.d_S_00 = self.dest.get_carray("S_00")
        self.d_S_01 = self.dest.get_carray("S_01")

        self.d_S_10 = self.dest.get_carray("S_10")
        self.d_S_11 = self.dest.get_carray("S_11")

        # setup the velocity gradient tensor arrays
        self.d_v_00 = self.dest.get_carray("v_00")
        self.d_v_01 = self.dest.get_carray("v_01")

        self.d_v_10 = self.dest.get_carray("v_10")
        self.d_v_11 = self.dest.get_carray("v_11")

    def setup_calc_updates(self, calc):
        r""" Setup the updates for the stress variable.

        The stress tensor is written as:

        .. math::

        \sigma^{ij} = -P\delta^{ij} + S^{ij}

        S is the deviatoric component of the stress tensor. It's nine
        components are devined as:

        S_00, S_01, S_02
        S_10, S_11, S_12
        S_20, S_21, S_22

        This deviatoric stress component will have to be integrated
        and the corresponding integrator variables are:

        Initial props:
        S_000, S_010, S_020
        S_100, S_110, S_120
        S_200, S_210, S_220

        Step props:
        S_00_a_1, S_01_a_1, S_02_a_1
        S_10_a_1, S_11_a_1, S_12_a_1
        S_20_a_1, S_21_a_1, S_22_a_1

        """
        calc.tensor_eval = True

        updates = []
        for i in range(2):
            for j in range(2):
                updates.append( "S_" + str(i) + str(j) )

        return updates
        
    cpdef tensor_eval(self, KernelBase kernel,
                      DoubleArray output1, DoubleArray output2,
                      DoubleArray output3, DoubleArray output4,
                      DoubleArray output5, DoubleArray output6,
                      DoubleArray output7, DoubleArray output8,
                      DoubleArray output9):

        """ Evaluate the nine components the deviatoric stress rate equation """ 
        
        # get the tag array pointer
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        # perform any iteration specific setup
        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        cdef size_t a

        cdef double result[9]

        for a in range(np):
            if tag_arr.data[a] == LocalReal:
                self.eval_single(a, kernel, result)

                output1.data[a] += result[0]
                output2.data[a] += result[1]

                output3.data[a] += result[2]
                output4.data[a] += result[3]

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):

        result[0] = result[1] = result[2] = 0.0
        result[3] = result[4] = result[5] = 0.0
        result[6] = result[7] = result[8] = 0.0

        cdef double v_00 = self.d_v_00.data[dest_pid]
        cdef double v_01 = self.d_v_01.data[dest_pid]

        cdef double v_10 = self.d_v_10.data[dest_pid]
        cdef double v_11 = self.d_v_11.data[dest_pid]

        cdef double s_00 = self.d_S_00.data[dest_pid]
        cdef double s_01 = self.d_S_01.data[dest_pid]

        cdef double s_10 = self.d_S_10.data[dest_pid]
        cdef double s_11 = self.d_S_11.data[dest_pid]

        # strain rate tensor is symmetric
        cdef double eps_00 = v_00
        cdef double eps_01 = 0.5 * (v_01 + v_10)

        cdef double eps_10 = eps_01
        cdef double eps_11 = v_11

        # rotation tensor is asymmetric
        cdef double omega_01 = 0.5 * (v_01 - v_10)
        cdef double omega_10 = -omega_01

        cdef double tmp = 2.0*self.shear_mod
        cdef double trace = 1.0/3.0 * (eps_00 + eps_11)

        # S_00
        result[0] = tmp*( eps_00 - trace ) + \
                    ( s_01*omega_01 ) + ( s_10*omega_01 )

        # S_01
        result[1] = tmp*(eps_01) + \
                    ( s_00*omega_10 ) + ( s_11*omega_01 )        

        # S_10
        #result[3] = tmp*(eps_10) + \
        #            ( s_11*omega_01 ) + ( s_00*omega_10 )
        result[2] = result[1]
        
        # S_11
        result[3] = tmp*( eps_11 - trace ) + \
                    ( s_10*omega_10 ) + ( s_01*omega_10 )


#############################################################################
# `MomentumEquationWithStress` class
#############################################################################
cdef class MomentumEquationWithStress2D(SPHFunctionParticle):
    r""" Evaluate the momentum equation:

    :math:`$
    \frac{D\vec{v_a}^i}{Dt} = \sum_b
    m_b\left(\frac{\sigma_a^{ij}}{\rho_a^2} +
    \frac{\sigma_b^{ij}}{\rho_b^2} \right)\nabla_a\,W_{ab}$`

    Artificial stress to remove the tension instability is added to
    the momentum equation as described in `SPH elastic dynamics` by
    J.P. Gray and J.J. Moaghan and R.P. Swift, Computer Methods in
    Applied Mechanical Engineering. vol 190 (2001) pp 6641 - 6662

    """

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True,
                 deltap=None,
                 double n=1,
                 **kwargs):
        """ Constructor.

        Parameters:
        -----------
        
        source, dest : ParticleArray

        deltap : double
            The reference initial particle spacing used to compute the
            artificial stress term: W(q)/W(deltap)

        n : double
            The sensitivity parameter for the artificial stress term.

        epsp, epsm : double
            Factors to govern the magnitude of artificial stress when the
            stress is positive (p) and negative (m)

        fac : double
            Factor to stabilize the computation of theta

        """

        self.with_correction = False

        if deltap is not None:
            self.deltap = deltap
            if deltap < 0:
                raise RuntimeError("deltap cannot be negative!")
            
            self.with_correction = True

        self.n = n

        self.id = 'momentumequationwithstress2d'
        self.tag = "velocity"

        # check that the deviatoric stress components are defined. If
        # not, add them.
        stress_props = ["S_00", "S_01",
                                 "S_11"]

        dest_props = dest.properties.keys()
        for prop in stress_props:
            if not prop in dest_props:
                dest.add_property( dict(name=prop) )

        src_props = source.properties.keys()
        for prop in stress_props:
            if not prop in src_props:
                source.add_property( dict(name=prop) )

        # Ensure that the artificial stress properties are defined
        artificial_stress = ["R_00", "R_01",
                                     "R_11"]
        for prop in artificial_stress:
            if not prop in dest_props:
                msg = "Property %s not present in array %s"%(prop,
                                                             dest.name)
                raise RuntimeError(msg)
            if not prop in src_props:
                msg = "Property %s not present in array %s"%(prop,
                                                             source.name)
                raise RuntimeError(msg)
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)
    def set_src_dst_reads(self):
        pass

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)
        
        # setup the deviatoric stress components (symmetric so 01 = 10)
        self.d_S_00 = self.dest.get_carray("S_00")
        self.d_S_01 = self.dest.get_carray("S_01")
        self.d_S_10 = self.dest.get_carray("S_01")
        self.d_S_11 = self.dest.get_carray("S_11")

        self.s_S_00 = self.source.get_carray("S_00")
        self.s_S_01 = self.source.get_carray("S_01")
        self.s_S_10 = self.source.get_carray("S_01")
        self.s_S_11 = self.source.get_carray("S_11")

        # setup the artificial stress components ( symmetric so 01 = 10 )
        self.d_R_00 = self.dest.get_carray("R_00")
        self.d_R_01 = self.dest.get_carray("R_01")
        self.d_R_10 = self.dest.get_carray("R_01")
        self.d_R_11 = self.dest.get_carray("R_11")

        self.s_R_00 = self.source.get_carray("R_00")
        self.s_R_01 = self.source.get_carray("R_01")
        self.s_R_10 = self.source.get_carray("R_01")
        self.s_R_11 = self.source.get_carray("R_11")

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):        

        cdef double mb = self.s_m.data[source_pid]
        
        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        cdef double hab = 0.5 * (ha + hb)

        cdef double pa = self.d_p.data[dest_pid]
        cdef double pb = self.s_p.data[source_pid]

        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double rhoa2 = 1.0/(rhoa*rhoa)
        cdef double rhob2 = 1.0/(rhob*rhob)

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef double s_00a = self.d_S_00.data[dest_pid]
        cdef double s_01a = self.d_S_01.data[dest_pid]
        cdef double s_10a = self.d_S_10.data[dest_pid]
        cdef double s_11a = self.d_S_11.data[dest_pid]

        cdef double s_00b = self.s_S_00.data[source_pid]
        cdef double s_01b = self.s_S_01.data[source_pid]
        cdef double s_10b = self.s_S_10.data[source_pid]
        cdef double s_11b = self.s_S_11.data[source_pid]

        cdef cPoint grad, grada, gradb
        cdef double ax, ay

        cdef double w, wdeltap, fab
        cdef cPoint _ra, _rb

        # corrections in the original frame
        cdef double R_00a, R_01a, R_11a
        cdef double R_00b, R_01b, R_11b

        # artificial stress terms to be added to the accelerations
        cdef double artificial_stress_00 = 0.0
        cdef double artificial_stress_01 = 0.0
        cdef double artificial_stress_11 = 0.0
        
        # Add the pressure to the deviatoric stresses
        s_00a = s_00a - pa
        s_00b = s_00b - pb

        s_11a = s_11a - pa
        s_11b = s_11b - pb

        if self.with_correction:

            R_00a = self.d_R_00.data[dest_pid]
            R_01a = self.d_R_01.data[dest_pid]
            R_10a = self.d_R_10.data[dest_pid]
            R_11a = self.d_R_11.data[dest_pid]

            R_00b = self.s_R_00.data[source_pid]
            R_01b = self.s_R_01.data[source_pid]
            R_10b = self.s_R_10.data[source_pid]
            R_11b = self.s_R_11.data[source_pid]

            # compute the correction term W(q)/W(deltap)
            _ra = cPoint_new(self.deltap, 0.0, 0.0)
            _rb = cPoint_new(0.0, 0.0, 0.0)
            
            wdeltap = kernel.function(_ra, _rb, hab)
            w = kernel.function(self._dst, self._src, hab)

            fab = w/wdeltap
            fab = pow(fab, self.n)

            artificial_stress_00 = fab * (R_00a + R_00b)
            artificial_stress_01 = fab * (R_01a + R_01b)
            artificial_stress_11 = fab * (R_11a + R_11b)

            # # evaluate the artificial stress for `a`
            # theta = get_principal_stress_angle2D(s_00a, s_01a, s_11a,
            #                                      self.theta_factor)
            # cos_theta = cos(theta)
            # sin_theta = sin(theta)

            # cos_theta2 = cos_theta * cos_theta
            # sin_theta2 = sin_theta * sin_theta

            # sig_00 = cos_theta2*s_00a + 2.0*cos_theta*sin_theta*s_01a + \
            #          sin_theta2*s_11a

            # sig_11 = sin_theta2*s_00a - 2.0*cos_theta*sin_theta*s_01a + \
            #          cos_theta2*s_11a

            # R_00 = 0.0
            # if sig_00 > 0:
            #     R_00 = -self.epsp * sig_00 * rhoa2

            # R_11 = 0.0
            # if sig_11 > 0:
            #     R_11 = -self.epsp * sig_11 * rhoa2

            # R_00a = cos_theta2*R_00 + sin_theta2*R_11
            # R_11a = sin_theta2*R_00 + cos_theta2*R_11

            # R_01a = cos_theta*sin_theta*(R_00 - R_11)

            # # evaluate the artificial stress for `b`
            # theta = get_principal_stress_angle2D(s_00b, s_01b, s_11b,
            #                                      self.theta_factor)
            # cos_theta = cos(theta)
            # sin_theta = sin(theta)

            # cos_theta2 = cos_theta*cos_theta
            # sin_theta2 = sin_theta*sin_theta

            # sig_00 = cos_theta2*s_00b + 2.0*cos_theta*sin_theta*s_01b + \
            #          sin_theta2*s_11b

            # sig_11 = sin_theta2*s_00b - 2.0*cos_theta*sin_theta*s_01b + \
            #          cos_theta2*s_11b

            # R_00 = 0.0
            # if sig_00 > 0:
            #     R_00 = -self.epsp * sig_00 * rhob2

            # R_11 = 0.0
            # if sig_11 > 0:
            #     R_11 = -self.epsp * sig_11 * rhob2

            # R_00b = cos_theta2*R_00 + sin_theta2*R_11
            # R_11b = sin_theta2*R_00 + cos_theta2*R_11

            # R_01b = cos_theta*sin_theta*(R_00 - R_11)
        
        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5
        else:
            grad = kernel.gradient( self._dst, self._src, hab )

        # compute the accelerations
        ax = mb*( s_00a*rhoa2 + s_00b*rhob2 ) * grad.x + \
             mb*( s_01a*rhoa2 + s_01b*rhob2 ) * grad.y

        ay = mb*( s_10a*rhoa2 + s_10b*rhob2 ) * grad.x + \
             mb*( s_11a*rhoa2 + s_11b*rhob2 ) * grad.y

        # add the artificial stress terms (default: 0)
        ax = ax + mb*( artificial_stress_00*grad.x + \
                       artificial_stress_01*grad.y )

        ay = ay + mb*( artificial_stress_01*grad.x + \
                       artificial_stress_11*grad.y )

        nr[0] += ax
        nr[1] += ay

#############################################################################
# `EnergyEquationWithStress2D` class
#############################################################################
cdef class EnergyEquationWithStress2D(SPHFunctionParticle):
    r""" Evaluate the energy equation.

    :math:`$
    \frac{D\vec{v_a}^i}{Dt} = \sum_b
    m_b\left(\frac{\sigma_a^{ij}}{\rho_a^2} +
    \frac{\sigma_b^{ij}}{\rho_b^2} \right)\nabla_a\,W_{ab}$`

    Artificial stress to remove the tension instability is added to
    the momentum equation as described in `SPH elastic dynamics` by
    J.P. Gray and J.J. Moaghan and R.P. Swift, Computer Methods in
    Applied Mechanical Engineering. vol 190 (2001) pp 6641 - 6662

    """

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True,
                 deltap=None,
                 double n=1,
                 **kwargs):
        """ Constructor.

        Parameters:
        -----------
        
        source, dest : ParticleArray

        deltap : double
            The reference initial particle spacing used to compute the
            artificial stress term: W(q)/W(deltap)

        n : double
            The sensitivity parameter for the artificial stress term.

        epsp, epsm : double
            Factors to govern the magnitude of artificial stress when the
            stress is positive (p) and negative (m)

        fac : double
            Factor to stabilize the computation of theta

        """

        self.with_correction = False

        if deltap is not None:
            self.deltap = deltap
            if deltap < 0:
                raise RuntimeError("deltap cannot be negative!")
            
            self.with_correction = True

        self.n = n

        self.id = 'energyequationwithstress2d'
        self.tag = "velocity"

        # check that the deviatoric stress components are defined. If
        # not, add them.
        stress_props = ["S_00", "S_01",
                                 "S_11"]

        dest_props = dest.properties.keys()
        for prop in stress_props:
            if not prop in dest_props:
                dest.add_property( dict(name=prop) )

        src_props = source.properties.keys()
        for prop in stress_props:
            if not prop in src_props:
                source.add_property( dict(name=prop) )

        # Ensure that the artificial stress properties are defined
        artificial_stress = ["R_00", "R_01",
                                     "R_11"]
        for prop in artificial_stress:
            if not prop in dest_props:
                msg = "Property %s not present in array %s"%(prop,
                                                             dest.name)
                raise RuntimeError(msg)
            if not prop in src_props:
                msg = "Property %s not present in array %s"%(prop,
                                                             source.name)
                raise RuntimeError(msg)
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)
    def set_src_dst_reads(self):
        pass

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)
        
        # setup the deviatoric stress components (symmetric so 01 = 10)
        self.d_S_00 = self.dest.get_carray("S_00")
        self.d_S_01 = self.dest.get_carray("S_01")
        self.d_S_10 = self.dest.get_carray("S_01")
        self.d_S_11 = self.dest.get_carray("S_11")

        self.s_S_00 = self.source.get_carray("S_00")
        self.s_S_01 = self.source.get_carray("S_01")
        self.s_S_10 = self.source.get_carray("S_01")
        self.s_S_11 = self.source.get_carray("S_11")

        # setup the artificial stress components ( symmetric so 01 = 10 )
        self.d_R_00 = self.dest.get_carray("R_00")
        self.d_R_01 = self.dest.get_carray("R_01")
        self.d_R_10 = self.dest.get_carray("R_01")
        self.d_R_11 = self.dest.get_carray("R_11")

        self.s_R_00 = self.source.get_carray("R_00")
        self.s_R_01 = self.source.get_carray("R_01")
        self.s_R_10 = self.source.get_carray("R_01")
        self.s_R_11 = self.source.get_carray("R_11")

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):        

        cdef double mb = self.s_m.data[source_pid]
        
        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        cdef double hab = 0.5 * (ha + hb)

        cdef double pa = self.d_p.data[dest_pid]
        cdef double pb = self.s_p.data[source_pid]

        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double rhoa2 = 1.0/(rhoa*rhoa)
        cdef double rhob2 = 1.0/(rhob*rhob)

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef double s_00a = -self.d_S_00.data[dest_pid]
        cdef double s_01a = -self.d_S_01.data[dest_pid]
        cdef double s_10a = -self.d_S_10.data[dest_pid]
        cdef double s_11a = -self.d_S_11.data[dest_pid]

        cdef double s_00b = -self.s_S_00.data[source_pid]
        cdef double s_01b = -self.s_S_01.data[source_pid]
        cdef double s_10b = -self.s_S_10.data[source_pid]
        cdef double s_11b = -self.s_S_11.data[source_pid]

        cdef cPoint grad, grada, gradb
        cdef cPoint vab
        cdef double a_e

        cdef double w, wdeltap, fab
        cdef cPoint _ra, _rb

        # relative velocity vector
        vab.x = self.d_u.data[dest_pid]-self.s_u.data[source_pid]
        vab.y = self.d_v.data[dest_pid]-self.s_v.data[source_pid]
        vab.z = self.d_w.data[dest_pid]-self.s_w.data[source_pid]

        # corrections in the original frame
        cdef double R_00a, R_01a, R_11a
        cdef double R_00b, R_01b, R_11b

        # artificial stress terms to be added to the accelerations
        cdef double artificial_stress_00 = 0.0
        cdef double artificial_stress_01 = 0.0
        cdef double artificial_stress_11 = 0.0
        
        # Add the pressure to the deviatoric stresses
        s_00a = s_00a + pa
        s_00b = s_00b + pb

        s_11a = s_11a + pa
        s_11b = s_11b + pb

        if self.with_correction:

            R_00a = self.d_R_00.data[dest_pid]
            R_01a = self.d_R_01.data[dest_pid]
            R_10a = self.d_R_10.data[dest_pid]
            R_11a = self.d_R_11.data[dest_pid]

            R_00b = self.s_R_00.data[source_pid]
            R_01b = self.s_R_01.data[source_pid]
            R_10b = self.s_R_10.data[source_pid]
            R_11b = self.s_R_11.data[source_pid]

            # compute the correction term W(q)/W(deltap)
            _ra = cPoint_new(self.deltap, 0.0, 0.0)
            _rb = cPoint_new(0.0, 0.0, 0.0)
            
            wdeltap = kernel.function(_ra, _rb, hab)
            w = kernel.function(self._dst, self._src, hab)

            fab = w/wdeltap
            fab = pow(fab, self.n)

            artificial_stress_00 = fab * (R_00a + R_00b)
            artificial_stress_01 = fab * (R_01a + R_01b)
            artificial_stress_11 = fab * (R_11a + R_11b)

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5
        else:
            grad = kernel.gradient( self._dst, self._src, hab )

        # compute the accelerations
        a_e = 0.5*mb*( vab.x*(s_00a*rhoa2 + s_00b*rhob2) * grad.x + \
                       vab.x*(s_01a*rhoa2 + s_01b*rhob2) * grad.y + \
                       vab.y*(s_01a*rhoa2 + s_01b*rhob2) * grad.x + \
                       vab.y*(s_11a*rhoa2 + s_11b*rhob2) * grad.y )
        
        # add the artificial stress terms (default: 0)
        a_e = a_e + 0.5*mb*( vab.x*artificial_stress_00*grad.x +
                             vab.x*artificial_stress_01*grad.y + \
                             vab.y*artificial_stress_01*grad.x + \
                             vab.y*artificial_stress_11*grad.y )

        nr[0] += a_e

#############################################################################
# `VonMisesPlasticity2D` class
#############################################################################
cdef class VonMisesPlasticity2D(SPHFunction):
    r""" Evaluate the energy equation.

    :math:`$
    \frac{D\vec{v_a}^i}{Dt} = \sum_b
    m_b\left(\frac{\sigma_a^{ij}}{\rho_a^2} +
    \frac{\sigma_b^{ij}}{\rho_b^2} \right)\nabla_a\,W_{ab}$`

    Artificial stress to remove the tension instability is added to
    the momentum equation as described in `SPH elastic dynamics` by
    J.P. Gray and J.J. Moaghan and R.P. Swift, Computer Methods in
    Applied Mechanical Engineering. vol 190 (2001) pp 6641 - 6662

    """

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 double flow_stress,
                 bint setup_arrays=True,
                 **kwargs):
        """ Constructor.

        Parameters:
        -----------
        
        source, dest : ParticleArray

        deltap : double
            The reference initial particle spacing used to compute the
            artificial stress term: W(q)/W(deltap)

        n : double
            The sensitivity parameter for the artificial stress term.

        epsp, epsm : double
            Factors to govern the magnitude of artificial stress when the
            stress is positive (p) and negative (m)

        fac : double
            Factor to stabilize the computation of theta

        """

        self.flow_stress = flow_stress
        self.fac = sqrt(flow_stress/3.0)

        self.id = 'vonmisesplasticity2d'
        self.tag = "velocity"

        # check that the deviatoric stress components are defined. If
        # not, add them.
        stress_props = ["S_00", "S_01",
                                 "S_11"]

        dest_props = dest.properties.keys()
        for prop in stress_props:
            if not prop in dest_props:
                msg = "%s adding property %s to array `%s`"%(self.id,
                                                             prop,
                                                             dest.name)
                print msg
                dest.add_property( dict(name=prop) )

        SPHFunction.__init__(self, source, dest, setup_arrays,
                             **kwargs)
        
    def set_src_dst_reads(self):
        pass

    cpdef setup_arrays(self):
        SPHFunctionParticle.setup_arrays(self)
        
        # setup the deviatoric stress components (symmetric so 01 = 10)
        self.d_S_00 = self.dest.get_carray("S_00")
        self.d_S_01 = self.dest.get_carray("S_01")
        self.d_S_10 = self.dest.get_carray("S_01")
        self.d_S_11 = self.dest.get_carray("S_11")

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        """ Perform the gravity force computation """

        cdef double s_00 = self.d_S_00.data[dest_pid]
        cdef double s_01 = self.d_S_01.data[dest_pid]
        cdef double s_10 = self.d_S_10.data[dest_pid]
        cdef double s_11 = self.d_S_11.data[dest_pid]

        # calculate the stress invariant
        cdef double J2 = s_00*s_00 + 2*s_01*s_10 + s_11*s_11
        cdef double scale = 1.0
        
        if J2 > self.flow_stress:
            scale = self.fac/sqrt(J2)

        self.d_S_00.data[dest_pid] = s_00 * scale
        self.d_S_01.data[dest_pid] = s_01 * scale
        self.d_S_11.data[dest_pid] = s_11 * scale
