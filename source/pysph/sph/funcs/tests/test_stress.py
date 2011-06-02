
import unittest
import numpy
from numpy import random, linalg, empty, allclose, arange, zeros, ones, sqrt, dot
from numpy.linalg import norm

import matplotlib.pyplot as plt

from pysph.sph.funcs.stress_funcs import py_det, py_get_eigenvalues, py_get_eigenvector, get_K, py_transform2, py_transform2inv
from pysph.sph.funcs.stress_funcs import StressRateD, StressRateS, SimpleStressAcceleration
from pysph.sph.funcs import stress_funcs

from pysph.base.api import get_particle_array, Particles
import pysph.base.api as base
from pysph.base import kernels

from pysph.sph.api import SPHCalc

class TestLinalg(unittest.TestCase):
    
    def test_det(self):
        for i in range(10):
            d = random.random(3)
            s = random.random(3)
            
            m = empty((3,3))
            m.flat[::4] = d
            m[0,1] = m[1,0] = s[2]
            m[0,2] = m[2,0] = s[1]
            m[2,1] = m[1,2] = s[0]
            
            n = linalg.det(m)
            p = py_det(d, s)
            
            self.assertTrue(allclose(n, p), 'n=%s, p=%s\n%s; %s,%s'%(n,p,m,d,s))
    
    def test_eigenvalues(self):
        for i in range(10):
            d = random.random(3)
            s = random.random(3)
            
            m = empty((3,3))
            m.flat[::4] = d
            m[0,1] = m[1,0] = s[2]
            m[0,2] = m[2,0] = s[1]
            m[2,1] = m[1,2] = s[0]
            
            n = linalg.eigvals(m)
            p = py_get_eigenvalues(d, s)
            
            self.assertTrue(allclose(sorted(n), sorted(p)), 'n=%s, p=%s\n%s; %s,%s'%(n,p,m,d,s))

    def test_eigenvalues_diag(self):
        for i in range(-5,5):
            for j in range(-5,5):
                d = (i,j,0)
                s = (0,0,0)
                
                m = empty((3,3))
                m.flat[::4] = d
                m[0,1] = m[1,0] = s[2]
                m[0,2] = m[2,0] = s[1]
                m[2,1] = m[1,2] = s[0]
                
                n = linalg.eigvals(m)
                p = py_get_eigenvalues(d, s)
                
                self.assertTrue(allclose(sorted(n), sorted(p), atol=1e-7), 'n=%s, p=%s\n%s; %s,%s'%(n,p,m,d,s))

    def test_eigenvalues_side(self):
        for i in range(10):
            d = (0,0,0)
            s = random.random(3)
            
            m = empty((3,3))
            m.flat[::4] = d
            m[0,1] = m[1,0] = s[2]
            m[0,2] = m[2,0] = s[1]
            m[2,1] = m[1,2] = s[0]
            
            n = linalg.eigvals(m)
            p = py_get_eigenvalues(d, s)
            
            self.assertTrue(allclose(sorted(n,key=abs), sorted(p,key=abs)), 'n=%s, p=%s\n%s; %s,%s'%(n,p,m,d,s))

    def test_eigenvalues_singular(self):
        for i in range(10):
            s = (1,2,3)
            d = (0,2,4)
            
            m = empty((3,3))
            m.flat[::4] = d
            m[0,1] = m[1,0] = s[2]
            m[0,2] = m[2,0] = s[1]
            m[2,1] = m[1,2] = s[0]
            
            n = linalg.eigvals(m)
            p = py_get_eigenvalues(d, s)
            
            self.assertTrue(allclose(sorted(n,key=abs), sorted(p,key=abs)), 'n=%s, p=%s\n%s; %s,%s'%(n,p,m,d,s))

    def test_eigenvectors(self):
        # FIXME: may fail in case of repeated eigenvalues
        for i in range(10):
            d = random.random(3)
            s = random.random(3)
            
            m = empty((3,3))
            m.flat[::4] = d
            m[0,1] = m[1,0] = s[2]
            m[0,2] = m[2,0] = s[1]
            m[2,1] = m[1,2] = s[0]
            
            n, nv = linalg.eig(m)
            #p = sorted(py_get_eigenvalues(d, s))
            
            for i in range(3):
                pv = py_get_eigenvector(d, s, n[i])
                if pv[0]*nv[0,i] < 0:
                    pv = [-v for v in pv]
                self.assertTrue(allclose(nv[:,i], pv), 'n=%s, p=%s\n%s; %s,%s'%(nv[:,i],pv,m,d,s))
    
    def test_transform2(self):
        Ad = random.random(3)
        A = numpy.zeros((3,3))
        A.flat[::4] = Ad
        P = random.random((3,3))
        R = dot( dot(P.T,A), P)
        r = py_transform2(Ad, P)
        numpy.allclose(R, r)

    def test_transform2inv(self):
        Ad = random.random(3)
        A = numpy.zeros((3,3))
        A.flat[::4] = Ad
        P = random.random((3,3))
        R = dot( dot(P,A), P.T)
        r = py_transform2inv(Ad, P)
        numpy.allclose(R, r)

class TestStress1D(unittest.TestCase):
    def setUp(self):
        N = 10
        self.kernel = kernels.CubicSplineKernel(2)
        x = arange(N)
        z = y = zeros(N)
        mu = m = rho = ones(N)
        h = 2*m

        pa = get_particle_array(x=x, y=y, z=z, h=h, mu=mu, rho=rho, m=m,
                                tmp=z, tx=z, ty=m, tz=z, nx=m, ny=z,
                                nz=z, u=z, v=z, w=z,
                                ubar=z, vbar=z, wbar=z,
                                q=m, div=z, rhop=z, e_t=m*0.1,
                                rkpm_beta1=z,
                                rkpm_beta2=z,
                                rkpm_alpha=z,
                                rkpm_dbeta1dx=z,
                                rkpm_dbeta1dy=z,
                                rkpm_dbeta2dx=z,
                                rkpm_dbeta2dy=z)
        pa.constants['E'] = 1e9
        pa.constants['nu'] = 0.3
        pa.constants['G'] = pa.constants['E']/(2.0*1+pa.constants['nu'])
        pa.constants['K'] = get_K(pa.constants['G'], pa.constants['nu'])
        pa.constants['rho0'] = 1.0
        pa.constants['c_s'] = sqrt(pa.constants['K']/pa.constants['rho0'])
        print 'Number of particles: ', len(pa.x)
        self.pa = self.pb = pa

        self.particles = Particles(arrays=[pa, pa])
        
    def get_calc(self, func_getter):
        func = func_getter.get_func(self.pa, self.pb)
        calc = SPHCalc(self.particles, [self.pa], self.pb, self.kernel, [func],
                       ['_tmpx'])
        return calc

    def test_stress_rate_d(self):
        calc = self.get_calc(StressRateD)
        calc.sph()

    def test_stress_rate_s(self):
        calc = self.get_calc(StressRateS)
        calc.sph()
        

class TestStress2D(unittest.TestCase):
    def setUp(self):
        self.kernel = kernels.CubicSplineKernel(2)
        self.pa, self.pb = self.create_particles()

    def create_particles(self):
        #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
        dx = 0.002 # 2mm
        xl = -0.05
        L = 0.2
        H = 0.02
        hfac = 1.5
        x,y = numpy.mgrid[xl:L+dx/2:dx, -H/2:(H+dx)/2:dx]
        x = x.ravel()
        y = y.ravel()
        bdry = (x<dx/2)*1.0
        bdry_indices = numpy.flatnonzero(bdry)
        print 'num_particles', len(x)
        #print bdry, numpy.flatnonzero(bdry)
        m = numpy.ones_like(x)*dx*dx
        h = numpy.ones_like(x)*hfac*dx
        rho = numpy.ones_like(x)
        z = numpy.zeros_like(x)
        
        interior = (5*dx<x) * (x<L-5*dx) * (-H/2+3*dx<y) * (y<H/2-3*dx)
        self.interior_points = numpy.flatnonzero(interior)

        u = v = p = z
        cs = numpy.ones_like(x) * 10000.0
        
        pa = base.get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                     z=z,w=z,
                                     ubar=z, vbar=z, wbar=z,
                                     name='solid', type=1,
                                     sigma00=z, sigma11=z, sigma22=z,
                                     sigma01=z, sigma12=z, sigma02=z,
                                     MArtStress00=z, MArtStress11=z, MArtStress22=z,
                                     MArtStress01=z, MArtStress12=z, MArtStress02=z,
                                     bdry=bdry,
                                     interior=interior,
                                     tmp1=z, tmp2=z, tmp3=z,
                                     )
        
        pa.constants['E'] = 1e7
        pa.constants['nu'] = 0.25
        pa.constants['G'] = pa.constants['E']/(2.0*1+pa.constants['nu'])
        pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
        pa.constants['rho0'] = 1.0
        pa.constants['c_s'] = (pa.constants['K']/pa.constants['rho0'])**0.5
        pa.cs = numpy.ones_like(x) * pa.constants['c_s']
        print 'c_s:', pa.c_s
        print 'G:', pa.G/pa.c_s**2/pa.rho0
        pa.v *= pa.c_s
        print 'v_f:', pa.v[-1]/pa.c_s, '(%s)'%pa.v[-1]
        print 'T:', 2*numpy.pi/(pa.E*0.02**2*(1.875/0.2)**4/(12*pa.rho0*(1-pa.nu**2)))**0.5
        pa.set(idx=numpy.arange(len(pa.x)))
        print 'Number of particles: ', len(pa.x)
        
        # boundary particle array
        x, y = numpy.mgrid[xl:dx/2:dx, H/2+dx:H/2+3.5*dx:dx]
        x = x.ravel()
        y = y.ravel()
        x2, y2 = x, -y
        x = numpy.concatenate([x,x2])
        y = numpy.concatenate([y,y2])
        z = numpy.zeros_like(x)
        
        rho = numpy.ones_like(x)
        m = rho*dx*dx
        h = hfac*dx*rho
        
        pb = base.get_particle_array(x=x, x0=x, y=y, y0=y, m=m, rho=rho,
                                     h=h, p=z, u=z, v=z, z=z,w=z,
                                     ubar=z, vbar=z, wbar=z,
                                     name='bdry', type=1,
                                     sigma00=z, sigma11=z, sigma22=z,
                                     sigma01=z, sigma12=z, sigma02=z,
                                     MArtStress00=z, MArtStress11=z, MArtStress22=z,
                                     MArtStress01=z, MArtStress12=z, MArtStress02=z,
                                     tmp1=z, tmp2=z, tmp3=z,
                                     )
        
        pb.constants['E'] = 1e7
        pb.constants['nu'] = 0.25
        pb.constants['G'] = pb.constants['E']/(2.0*1+pb.constants['nu'])
        pb.constants['K'] = stress_funcs.get_K(pb.constants['G'], pb.constants['nu'])
        pb.constants['rho0'] = 1.0
        pb.constants['c_s'] = (pb.constants['K']/pb.constants['rho0'])**0.5
        pb.cs = numpy.ones_like(x) * pb.constants['c_s']
        
        return [pa, pb]

    def get_calc(self, func_getter, src=None, dst=None):
        if src is None:
            src = self.pa
        arrays = [src]
        if dst is None:
            dst = self.pa
        if dst is not src:
            arrays.append(dst)
        self.particles = Particles(arrays=arrays)
        func = func_getter.get_func(src, dst)
        calc = SPHCalc(self.particles, arrays, dst, self.kernel, [func],
                       ['_tmpx'])
        return calc

    def test_stress_rate_d(self):
        # TODO: also check the rotation rate effect
        dudx = 0.1
        self.pa.u = dudx * self.pa.x
        calc = self.get_calc(StressRateD.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        print self.pa.u
        print self.pa.tmp1, self.pa.tmp2
        dS00 = dudx*self.pa.G*4.0/3.0
        #dS00 = 2*self.pa.G*(dudx - dudx/3.)
        print dS00
        print norm(self.pa.tmp1-dS00), norm(self.pa.tmp1), norm(self.pa.tmp1-dS00)/norm(self.pa.tmp1)
        print norm(self.pa.tmp1[self.interior_points]-dS00), norm(self.pa.tmp1), norm(self.pa.tmp1[self.interior_points]-dS00)/norm(self.pa.tmp1[self.interior_points])

        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*dS00)
        plt.xlabel('x')
        plt.ylabel('$\\frac{dS}{dt}$')
        plt.savefig('dSdt00.svg')
        
    def test_stress_rate_s(self):
        # TODO: also check the rotation rate effect
        dvdx = 0.1
        self.pa.v = dvdx * self.pa.x
        calc = self.get_calc(StressRateS.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        print self.pa.v
        print self.pa.tmp1, self.pa.tmp2
        dS01 = dvdx*self.pa.G
        #dS00 = 2*self.pa.G*(dudx - dudx/3.)
        print dS01
        print norm(self.pa.tmp3-dS01), norm(self.pa.tmp3), norm(self.pa.tmp3-dS01)/norm(self.pa.tmp3)
        print norm(self.pa.tmp3[self.interior_points]-dS01), norm(self.pa.tmp3), norm(self.pa.tmp3[self.interior_points]-dS01)/norm(self.pa.tmp3[self.interior_points])

        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp3, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp3[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*dS01)
        plt.xlabel('x')
        plt.ylabel('$\\frac{dS}{dt}$')
        plt.savefig('dSdt01.svg')
        
        
    def test_stress_acc_d(self):
        ds00dx = 1e3
        self.pa.sigma00 = ds00dx * numpy.sin(self.pa.x/5*numpy.pi)
        self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(SimpleStressAcceleration)
        calc.sph('tmp1', 'tmp2', 'tmp3')
        print self.pa.sigma00, self.pa.sigma11, self.pa.sigma22
        print self.pa.sigma00 + self.pa.sigma11 + self.pa.sigma22
        
        print self.pa.tmp1, self.pa.tmp2
        ax = ds00dx
        print ax
        print norm(self.pa.tmp1-ax), norm(self.pa.tmp1), norm(self.pa.tmp1-ax)/norm(self.pa.tmp1)
        print norm(self.pa.tmp1[self.interior_points]-ax), norm(self.pa.tmp1[self.interior_points]), norm(self.pa.tmp1[self.interior_points]-ax)/norm(self.pa.tmp1[self.interior_points])

        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ax)
        plt.xlabel('x')
        plt.ylabel('$ax$')
        plt.savefig('S0011_ax.svg')

        ay = 0
        print ay
        print norm(self.pa.tmp2-ay), norm(self.pa.tmp2), norm(self.pa.tmp2-ay)/norm(self.pa.tmp2)
        print norm(self.pa.tmp2[self.interior_points]-ay), norm(self.pa.tmp2[self.interior_points]), norm(self.pa.tmp2[self.interior_points]-ay)/norm(self.pa.tmp2[self.interior_points])
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp2, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp2[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ay)
        plt.xlabel('x')
        plt.ylabel('$ay$')
        plt.savefig('S0011_ay.svg')


    def test_stress_acc_s(self):
        ds01dx = 1e3
        self.pa.sigma01 = ds01dx * (self.pa.x-max(self.pa.x))
        #self.pa.sigma01[:] = ds01dx
        #self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(SimpleStressAcceleration)
        calc.sph('tmp1', 'tmp2', 'tmp3')
        print self.pa.sigma00, self.pa.sigma11, self.pa.sigma22
        print self.pa.sigma00 + self.pa.sigma11 + self.pa.sigma22
        
        print self.pa.tmp1, self.pa.tmp2
        ax = 0
        print ax
        print norm(self.pa.tmp1-ax), norm(self.pa.tmp1), norm(self.pa.tmp1-ax)/norm(self.pa.tmp1)
        print norm(self.pa.tmp1[self.interior_points]-ax), norm(self.pa.tmp1[self.interior_points]), norm(self.pa.tmp1[self.interior_points]-ax)/norm(self.pa.tmp1[self.interior_points])

        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ax)
        plt.xlabel('x')
        plt.ylabel('$ax$')
        plt.savefig('S01_ax.svg')

        ay = ds01dx
        print ay
        print norm(self.pa.tmp2-ay), norm(self.pa.tmp2), norm(self.pa.tmp2-ay)/norm(self.pa.tmp2)
        print norm(self.pa.tmp2[self.interior_points]-ay), norm(self.pa.tmp2[self.interior_points]), norm(self.pa.tmp2[self.interior_points]-ay)/norm(self.pa.tmp2[self.interior_points])

        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp2, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp2[self.interior_points], '.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ay)
        plt.xlabel('x')
        plt.ylabel('$ay$')
        plt.savefig('S01_ay.svg')



    def test_monaghan_art_stress_d(self):
        ds00dx = 1e3
        self.pa.sigma00 = ds00dx * numpy.sin(10*self.pa.x*numpy.pi)
        #self.pa.sigma00 = ds00dx * self.pa.x
        #self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(stress_funcs.MonaghanArtStressD.withargs(eps=0.3))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        
        print self.pa.tmp1, self.pa.tmp2
        R = -0.3*self.pa.sigma00*(self.pa.sigma00>0)
        print R
        print norm(self.pa.tmp1-R), norm(self.pa.tmp1), norm(self.pa.tmp1-R)/norm(self.pa.tmp1)
        print norm((self.pa.tmp1-R)[self.interior_points]), norm(self.pa.tmp1[self.interior_points]), norm((self.pa.tmp1-R)[self.interior_points])/norm(self.pa.tmp1[self.interior_points])
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*R)
        plt.xlabel('x')
        plt.ylabel('$R$')
        plt.savefig('MartS00.svg')

        # ay = 0
        # print ay
        # print norm(self.pa.tmp2-ay), norm(self.pa.tmp2), norm(self.pa.tmp2-ay)/norm(self.pa.tmp2)
        # print norm(self.pa.tmp2[self.interior_points]-ay), norm(self.pa.tmp2[self.interior_points]), norm(self.pa.tmp2[self.interior_points]-ay)/norm(self.pa.tmp2[self.interior_points])
        # fig = plt.figure()
        # line, = plt.plot(self.pa.x, self.pa.tmp2, '.')
        # line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp2[self.interior_points], 'r.')
        # line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ay)
        # plt.xlabel('x')
        # plt.ylabel('$ay$')
        # plt.savefig('MartS11.svg')





    def _test_monaghan_art_stress(self):
        ds00dx = 1e3
        self.pa.sigma00 = ds00dx * numpy.sin(self.pa.x/5*numpy.pi)
        self.pa.sigma00 = ds00dx * self.pa.x
        #self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(stress_funcs.MonaghanArtStress)
        calc.sph('tmp1', 'tmp2', 'tmp3')
        print self.pa.sigma00, self.pa.sigma11, self.pa.sigma22
        print self.pa.sigma00 + self.pa.sigma11 + self.pa.sigma22
        
        print self.pa.tmp1, self.pa.tmp2
        ax = ds00dx
        print ax
        print norm(self.pa.tmp1-ax), norm(self.pa.tmp1), norm(self.pa.tmp1-ax)/norm(self.pa.tmp1)
        print norm(self.pa.tmp1[self.interior_points]-ax), norm(self.pa.tmp1[self.interior_points]), norm(self.pa.tmp1[self.interior_points]-ax)/norm(self.pa.tmp1[self.interior_points])
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ax)
        plt.xlabel('x')
        plt.ylabel('$ax$')
        plt.savefig('Mart_ax.svg')

        ay = 0
        print ay
        print norm(self.pa.tmp2-ay), norm(self.pa.tmp2), norm(self.pa.tmp2-ay)/norm(self.pa.tmp2)
        print norm(self.pa.tmp2[self.interior_points]-ay), norm(self.pa.tmp2[self.interior_points]), norm(self.pa.tmp2[self.interior_points]-ay)/norm(self.pa.tmp2[self.interior_points])
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp2, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp2[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ay)
        plt.xlabel('x')
        plt.ylabel('$ay$')
        plt.savefig('Mart_ay.svg')

        
if __name__ == '__main__':
    unittest.main()
