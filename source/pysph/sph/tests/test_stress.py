import nose
import unittest
import numpy
from numpy import random, linalg, empty, allclose, arange, zeros, ones, sqrt, dot
from numpy import sin, cos, sign, diag
from numpy.linalg import norm

import tempfile
import shutil

import matplotlib.pyplot as plt

from pysph.sph.funcs.linalg import py_det, py_get_eigenvalues, py_get_eigenvector, py_transform2, py_transform2inv, py_get_eigenvalvec
from pysph.sph.funcs.stress_funcs import StressRateD, StressRateS, SimpleStressAcceleration, get_K
from pysph.sph.funcs import stress_funcs

from pysph.solver.stress_solver import StressSolver
from pysph.solver.integrator import EulerIntegrator
from pysph.solver.application import Application

from pysph.base.api import get_particle_array, Particles
import pysph.base.api as base
from pysph.base import kernels

from pysph.sph.api import SPHCalc

def disabled(f):
    def _decorator(arg):
        print  arg,  " has been disabled"
    return _decorator


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

    def test_eigenvalvec(self):
        for i in range(10):
            d = random.random(3)
            s = random.random(3)
            
            m = empty((3,3))
            m.flat[::4] = d
            m[0,1] = m[1,0] = s[2]
            m[0,2] = m[2,0] = s[1]
            m[2,1] = m[1,2] = s[0]
            
            n, nv = linalg.eig(m)
            p = numpy.argsort(numpy.abs(n))[::-1]
            nv = nv[:,p]

            r, v = py_get_eigenvalvec(d, s)
            p = numpy.argsort(numpy.abs(r))[::-1]
            r = numpy.array(r)[p]
            v = v[:,p]
            
            for i in range(3):
                pv = v[:,i]
                if pv[0]*nv[0,i] < 0:
                    pv = [-t for t in pv]
                self.assertTrue(allclose(nv[:,i], pv), 'ev:%s(%s)&%s(%s),\n n=%s, p=%s\n%s; %s,%s;\npv:%s,\nnv%s:'%(r[i],r,n[i],n, nv[:,i],pv,m,d,s, pv, nv))


        # check for singular/repeated eval systems
        for d, s in [((1,1,1),(0,0,0)),
                     ((0,0,0),(0,0,0)),
                     ((1,2,3),(0,0,0)),
                     ((1e-1,2e-7,3e-9),(0,0,0)),
                     ((1,1,1),(1e-5,0,0)),
                     ((1,1,1),(1e-8,1e-8,1e-8)),
                     ((0,0,0),(0,1e-9,0)),
                     ((1,2,3),(0,0,1e-9)),
                     ((1e-1,2e-7,3e-9),(0,1,0)),
                     ((0.07807302,1.13970482,1.80565157),(1.41514252,-0.27644203,-0.18357224)),
                     ]:
            d = numpy.array(d,dtype='float')
            s = numpy.array(s,dtype='float')
            
            m = empty((3,3))
            m.flat[::4] = d
            m[0,1] = m[1,0] = s[2]
            m[0,2] = m[2,0] = s[1]
            m[2,1] = m[1,2] = s[0]
            
            n, nv = linalg.eig(m)
            p = numpy.argsort(n)[::-1]
            n = n[p]
            nv = nv[:,p]

            r, v = py_get_eigenvalvec(d, s)
            p = numpy.argsort(r)[::-1]
            r = numpy.array(r)[p]
            v = v[:,p]

            self.assertTrue(allclose(n,r), 'n:%s,r:%s\nm:%s'%(n,r,m))
            if norm((n[:2]-n[-1:])*(n[2]-n[0]))==0:
                # repeated eigenvalues:
                # FIXME: check eigenvectors also for repeated eigenvalues
                continue

            self.assertTrue(allclose(numpy.dot(v.T, numpy.dot(m, v)), diag(r)))

            # this will not work for numpy 1.3.0
            #self.assertTrue(allclose(v.T.dot(m.dot(v)), diag(r)))

            for i in range(3):
                # check for normalization
                self.assertTrue(allclose(sum(v[:,i]**2),1))

            self.assertTrue(allclose(r, n))

            # for i in range(3):
            #     pv = v[:,i]
            #     if numpy.all(pv*nv[:,i] <= 0):
            #         pv = [-t for t in pv]
            #     self.assertTrue(allclose(nv[:,i], pv), 'ev:%s(%s)&%s(%s),\n '+
            #        'n=%s, p=%s\n%s; %s,%s;\npv:%s,\nnv:%s'%(r[i],r,n[i],n, '+
            #        'nv[:,i],pv,m,d,s, v, nv))


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
        #print 'Number of particles: ', len(pa.x)
        self.pa = self.pb = pa

        self.particles = Particles(arrays=[pa, pa])
        
    def get_calc(self, func_getter):
        func = func_getter.get_func(self.pa, self.pb)
        calc = SPHCalc(self.particles, [self.pa], self.pb, self.kernel, [func],
                       ['_tmpx'])
        return calc

    def test_stress_rate_d(self):
        raise nose.SkipTest('This test segfaults!')
        calc = self.get_calc(StressRateD)
        calc.sph()

    def test_stress_rate_s(self):
        raise nose.SkipTest('This test segfaults!')
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
        #print 'num_particles', len(x)
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
        #print 'c_s:', pa.c_s
        #print 'G:', pa.G/pa.c_s**2/pa.rho0
        pa.v *= pa.c_s
        #print 'v_f:', pa.v[-1]/pa.c_s, '(%s)'%pa.v[-1]
        #print 'T:', 2*numpy.pi/(pa.E*0.02**2*(1.875/0.2)**4/(12*pa.rho0*(1-pa.nu**2)))**0.5
        pa.set(idx=numpy.arange(len(pa.x)))
        #print 'Number of particles: ', len(pa.x)
        
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
        #print self.pa.u
        #print self.pa.tmp1, self.pa.tmp2
        dS00 = dudx*self.pa.G*4.0/3.0
        
        #print dS00
        #print norm(self.pa.tmp1-dS00), norm(self.pa.tmp1), norm(self.pa.tmp1-dS00)/norm(self.pa.tmp1)
        #print norm(self.pa.tmp1[self.interior_points]-dS00), norm(self.pa.tmp1), norm(self.pa.tmp1[self.interior_points]-dS00)/norm(self.pa.tmp1[self.interior_points])

        """
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*dS00)
        plt.xlabel('x')
        plt.ylabel('$\\frac{dS}{dt}$')
        plt.savefig('dSdt00.svg')
        """
        
    def test_stress_rate_s(self):
        # TODO: also check the rotation rate effect
        dvdx = 0.1
        self.pa.v = dvdx * self.pa.x
        calc = self.get_calc(StressRateS.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        #print self.pa.v
        #print self.pa.tmp1, self.pa.tmp2
        dS01 = dvdx*self.pa.G
        
        #print dS01
        #print norm(self.pa.tmp3-dS01), norm(self.pa.tmp3), norm(self.pa.tmp3-dS01)/norm(self.pa.tmp3)
        #print norm(self.pa.tmp3[self.interior_points]-dS01), norm(self.pa.tmp3), norm(self.pa.tmp3[self.interior_points]-dS01)/norm(self.pa.tmp3[self.interior_points])

        """
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp3, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp3[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*dS01)
        plt.xlabel('x')
        plt.ylabel('$\\frac{dS}{dt}$')
        plt.savefig('dSdt01.svg')
        """
        
    def test_stress_acc_d(self):
        ds00dx = 1e3
        self.pa.sigma00 = ds00dx * numpy.sin(self.pa.x/5*numpy.pi)
        self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(SimpleStressAcceleration)
        calc.sph('tmp1', 'tmp2', 'tmp3')
        #print self.pa.sigma00, self.pa.sigma11, self.pa.sigma22
        #print self.pa.sigma00 + self.pa.sigma11 + self.pa.sigma22
        
        #print self.pa.tmp1, self.pa.tmp2
        ax = ds00dx
        #print ax
        #print norm(self.pa.tmp1-ax), norm(self.pa.tmp1), norm(self.pa.tmp1-ax)/norm(self.pa.tmp1)
        #print norm(self.pa.tmp1[self.interior_points]-ax), norm(self.pa.tmp1[self.interior_points]), norm(self.pa.tmp1[self.interior_points]-ax)/norm(self.pa.tmp1[self.interior_points])
        """
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ax)
        plt.xlabel('x')
        plt.ylabel('$ax$')
        plt.savefig('S0011_ax.svg')

        ay = 0
        #print ay
        #print norm(self.pa.tmp2-ay), norm(self.pa.tmp2), norm(self.pa.tmp2-ay)/norm(self.pa.tmp2)
        #print norm(self.pa.tmp2[self.interior_points]-ay), norm(self.pa.tmp2[self.interior_points]), norm(self.pa.tmp2[self.interior_points]-ay)/norm(self.pa.tmp2[self.interior_points])
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp2, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp2[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ay)
        plt.xlabel('x')
        plt.ylabel('$ay$')
        plt.savefig('S0011_ay.svg')
        """

    def test_stress_acc_s(self):
        ds01dx = 1e3
        self.pa.sigma01 = ds01dx * (self.pa.x-max(self.pa.x))
        #self.pa.sigma01[:] = ds01dx
        #self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(SimpleStressAcceleration)
        calc.sph('tmp1', 'tmp2', 'tmp3')
        #print self.pa.sigma00, self.pa.sigma11, self.pa.sigma22
        #print self.pa.sigma00 + self.pa.sigma11 + self.pa.sigma22
        
        #print self.pa.tmp1, self.pa.tmp2
        ax = 0
        #print ax
        #print norm(self.pa.tmp1-ax), norm(self.pa.tmp1), norm(self.pa.tmp1-ax)/norm(self.pa.tmp1)
        #print norm(self.pa.tmp1[self.interior_points]-ax), norm(self.pa.tmp1[self.interior_points]), norm(self.pa.tmp1[self.interior_points]-ax)/norm(self.pa.tmp1[self.interior_points])
        """
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ax)
        plt.xlabel('x')
        plt.ylabel('$ax$')
        plt.savefig('S01_ax.svg')

        ay = ds01dx
        #print ay
        #print norm(self.pa.tmp2-ay), norm(self.pa.tmp2), norm(self.pa.tmp2-ay)/norm(self.pa.tmp2)
        #print norm(self.pa.tmp2[self.interior_points]-ay), norm(self.pa.tmp2[self.interior_points]), norm(self.pa.tmp2[self.interior_points]-ay)/norm(self.pa.tmp2[self.interior_points])

        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp2, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp2[self.interior_points], '.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ay)
        plt.xlabel('x')
        plt.ylabel('$ay$')
        plt.savefig('S01_ay.svg')
        """


    def test_monaghan_art_stress_d(self):
        ds00dx = 1e3
        self.pa.sigma00 = ds00dx * numpy.sin(10*self.pa.x*numpy.pi)
        #self.pa.sigma00 = ds00dx * self.pa.x
        #self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(stress_funcs.MonaghanArtStressD.withargs(eps=0.3))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        
        #print self.pa.tmp1, self.pa.tmp2
        R = -0.3*self.pa.sigma00*(self.pa.sigma00>0)
        #print R
        #print norm(self.pa.tmp1-R), norm(self.pa.tmp1), norm(self.pa.tmp1-R)/norm(self.pa.tmp1)
        #print norm((self.pa.tmp1-R)[self.interior_points]), norm(self.pa.tmp1[self.interior_points]), norm((self.pa.tmp1-R)[self.interior_points])/norm(self.pa.tmp1[self.interior_points])
        """
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*R)
        plt.xlabel('x')
        plt.ylabel('$R$')
        plt.savefig('MartS00.svg')
        """


    def test_monaghan_art_stress(self):
        ds00dx = 1e3
        self.pa.sigma00 = ds00dx * numpy.sin(self.pa.x/5*numpy.pi)
        self.pa.sigma00 = ds00dx * self.pa.x
        #self.pa.sigma11 = -self.pa.sigma00
        # trace of deviatoric stress should be zero
        calc = self.get_calc(stress_funcs.MonaghanArtificialStress)
        calc.sph('tmp1', 'tmp2', 'tmp3')
        #print self.pa.sigma00, self.pa.sigma11, self.pa.sigma22
        #print self.pa.sigma00 + self.pa.sigma11 + self.pa.sigma22
        
        """
        #print self.pa.tmp1, self.pa.tmp2
        ax = ds00dx
        #print ax
        #print norm(self.pa.tmp1-ax), norm(self.pa.tmp1), norm(self.pa.tmp1-ax)/norm(self.pa.tmp1)
        #print norm(self.pa.tmp1[self.interior_points]-ax), norm(self.pa.tmp1[self.interior_points]), norm(self.pa.tmp1[self.interior_points]-ax)/norm(self.pa.tmp1[self.interior_points])
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp1, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp1[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ax)
        plt.xlabel('x')
        plt.ylabel('$ax$')
        plt.savefig('Mart_ax.svg')

        ay = 0
        #print ay
        #print norm(self.pa.tmp2-ay), norm(self.pa.tmp2), norm(self.pa.tmp2-ay)/norm(self.pa.tmp2)
        #print norm(self.pa.tmp2[self.interior_points]-ay), norm(self.pa.tmp2[self.interior_points]), norm(self.pa.tmp2[self.interior_points]-ay)/norm(self.pa.tmp2[self.interior_points])
        fig = plt.figure()
        line, = plt.plot(self.pa.x, self.pa.tmp2, '.')
        line, = plt.plot(self.pa.x[self.interior_points], self.pa.tmp2[self.interior_points], 'r.')
        line, = plt.plot(self.pa.x, numpy.ones_like(self.pa.x)*ay)
        plt.xlabel('x')
        plt.ylabel('$ay$')
        plt.savefig('Mart_ay.svg')

        """


class TestStress3D(unittest.TestCase):
    def setUp(self):
        self.kernel = kernels.CubicSplineKernel(3)
        self.pa = self.create_particles()
        self.solver = solver = StressSolver(dim=3, integrator_type=EulerIntegrator,
                                            xsph=0.5, marts_eps=0.3, marts_n=4)
        solver.dt = 1e-7
        solver.tf = 1e-7
        self.output_dir = tempfile.mkdtemp()
        pa = self.create_particles()
        self.particles  = particles = base.Particles(arrays=[pa,])
        solver.setup(particles)

        #self.app = app = Application()
        #app.set_solver(solver, self.create_particles, False)
        #self.particles = app.particles

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def create_particles(self):
        #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
        n = 10
        self.l = l = 1.0
        self.dx = dx = 1.0/n
        # np = (n*n+1)**3
        
        x,y,z = numpy.mgrid[-l:l:dx, -l:l:dx, -l:l:dx]
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        self.idx = numpy.ones_like(x)
        self.idx *= (3-n)*dx<x
        self.idx *= (3-n)*dx<y
        self.idx *= (3-n)*dx<z
        self.idx *= x<(n-3)*dx
        self.idx *= y<(n-3)*dx
        self.idx *= z<(n-3)*dx
        self.idx = numpy.flatnonzero(self.idx)
        #print (2*n+1)**3, len(self.idx)

        # -1 <= f,g <= 1
        self.f = f = (sin(2*x/l) + cos(2*y/l) + z/l)/3.0
        self.fx = 2*cos(2*x/l)/l/3
        self.fy = -2*sin(2*y/l)/l/3
        self.fz = numpy.ones_like(self.f)/l/3

        self.g = g = sin(2*x/l) * cos(2*y/l) * z/l
        self.gx = 2*cos(2*x/l)/l * cos(2*y/l) * z/l
        self.gy = -2*sin(2*x/l)/l * sin(2*y/l) * z/l
        self.gz = 1/l * sin(2*x/l) * cos(2*y/l)

        self.h = (1+self.f*self.g)**0.5
        self.hx = (self.fx*self.g + self.f*self.gx)/2/self.h
        self.hy = (self.fy*self.g + self.f*self.gy)/2/self.h
        self.hz = (self.fz*self.g + self.f*self.gz)/2/self.h

        self.zero = zero = numpy.zeros_like(x)
        self.one = one = numpy.ones_like(x)

        m = one*dx*dx*dx
        h = 1.5*one*dx
        rho = one

        p = zero # will be computed
        cs = 10000.0 * one # will be set later

        u = v = w = zero

        sigma00 = zero
        sigma11 = zero
        sigma22 = zero
        sigma01 = zero
        sigma12 = zero
        sigma02 = zero

        pa = base.get_particle_array(x=x,y=y,z=z,m=m,rho=rho,h=h,p=p,u=u,v=v,w=w,
                                     ubar=zero, vbar=zero, wbar=zero,
                                     name='block', type=1,
                                     sigma00=sigma00,sigma11=sigma11,sigma22=sigma22,
                                     sigma01=sigma01,sigma12=sigma12,sigma02=sigma02,
                                     MArtStress00=zero, MArtStress11=zero,
                                     MArtStress22=zero,
                                     MArtStress01=zero, MArtStress12=zero,
                                     MArtStress02=zero,
                                     tmp1=zero, tmp2=zero, tmp3=zero,
                                     )
        
        pa.constants['E'] = 1e7
        pa.constants['nu'] = 0.3975
        pa.constants['G'] = pa.constants['E']/(2.0*1+pa.constants['nu'])
        pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
        pa.constants['rho0'] = 1.0
        pa.constants['c_s'] = (pa.constants['K']/pa.constants['rho0'])**0.5
        pa.cs = numpy.ones_like(x) * pa.constants['c_s']

        pa.set(idx=numpy.arange(len(pa.x)))
        
        return pa

    def set_all_props(self):
        pa = self.particles.arrays[0]
        f, g = self.f, self.g
        
        pa.rho = 1-g/10

        pa.u = f*g
        pa.v = f*f
        pa.w = (f+g)/2

        pa.sigma00 = f
        pa.sigma11 = g
        pa.sigma22 = (f+g)/2
        pa.sigma01 = f*f
        pa.sigma12 = g*g
        pa.sigma02 = f*g


    def get_calc(self, func_getter, src=None, dst=None):
        if src is None:
            src = self.pa
        arrays = [src]
        if dst is None:
            dst = self.pa
        if dst is not src:
            arrays.append(dst)
        func = func_getter.get_func(src, dst)
        calc = SPHCalc(self.particles, arrays, dst, self.kernel, [func],
                       ['_tmpx'])
        return calc

    def check(self, got, exp, rtol=1e-6, atol=1e-8):
        self.assertTrue(allclose(exp[self.idx], got[self.idx], rtol=rtol,atol=atol),
                        msg='expected:%s,\ngot:%s;\n%s, %s, %s'%(exp,got,
             norm((exp-got)[self.idx]), norm(got[self.idx]),
             norm((exp-got)[self.idx])/norm(got[self.idx])))

    
    def test_bulk_modulus_p_eqn(self):
        pa = self.pa
        pa.rho = 1-sin(2*pa.x/self.l)/1e3
        exp = (pa.rho-pa.rho0)*pa.c_s**2

        calc = self.get_calc(stress_funcs.BulkModulusPEqn.withargs())
        calc.sph('tmp1', 'tmp2', 'tmp3')
        got = pa.tmp1

        self.assertTrue(allclose(exp[self.idx], got[self.idx]), msg='expected:%s, got:%s'%(exp,got))

    def test_monaghan_art_stress_d_d(self):
        pa = self.pa
        pa.sigma00 = self.f
        exp1 = -0.3*pa.sigma00*(pa.sigma00>0)

        calc = self.get_calc(stress_funcs.MonaghanArtStressD.withargs(eps=0.3))
        calc.sph('tmp1', 'tmp2', 'tmp3')

        self.check(pa.tmp1, exp1)
        self.check(pa.tmp2, self.zero)
        self.check(pa.tmp3, self.zero)

    def test_monaghan_art_stress_d_p(self):
        pa = self.pa
        pa.p = self.f
        exp1 = 0.3*pa.p*(pa.p<0)

        calc = self.get_calc(stress_funcs.MonaghanArtStressD.withargs(eps=0.3))
        calc.sph('tmp1', 'tmp2', 'tmp3')

        got, exp = pa.tmp1, exp1
        self.assertTrue(allclose(exp[self.idx], got[self.idx]),
                        msg='expected:%s,\ngot:%s;\n%s, %s, %s'%(exp,got,
             norm((exp-got)[self.idx]), norm(got[self.idx]),
             norm((exp-got)[self.idx])/norm(got[self.idx])))

        got, exp = pa.tmp2, exp1
        self.assertTrue(allclose(exp[self.idx], got[self.idx]), msg='expected:%s, got:%s'%(exp,got))

        got, exp = pa.tmp3, exp1
        self.assertTrue(allclose(exp[self.idx], got[self.idx]), msg='expected:%s, got:%s'%(exp,got))

    def test_monaghan_art_stress_d_s(self):
        pa = self.pa
        # principal axes are 45 deg inclined to x-y axes
        pa.sigma01 = self.f
        exp1 = -0.3*pa.sigma01*(sign(pa.sigma01))/2

        calc = self.get_calc(stress_funcs.MonaghanArtStressD.withargs(eps=0.3))
        calc.sph('tmp1', 'tmp2', 'tmp3')

        got, exp = pa.tmp1, exp1
        self.assertTrue(allclose(exp[self.idx], got[self.idx]),
                        msg='expected:%s,\ngot:%s;\n%s, %s, %s'%(exp,got,
             norm((exp-got)[self.idx]), norm(got[self.idx]),
             norm((exp-got)[self.idx])/norm(got[self.idx])))

        got, exp = pa.tmp2, exp1
        self.assertTrue(allclose(exp[self.idx], got[self.idx]), msg='expected:%s, got:%s'%(exp,got))

        got, exp = pa.tmp3, self.zero
        self.assertTrue(allclose(exp[self.idx], got[self.idx]), msg='expected:%s, got:%s'%(exp,got))


    def test_monaghan_art_stress_acc(self):
        raise nose.SkipTest('Not implemented!')

    def test_stress_rate_d_1(self):
        # TODO: also check the rotation rate effect
        pa = self.pa
        pa.u = self.f
        e00 = self.fx
        e01 = self.fy/2
        e02 = self.fz/2
        exp1 = 2*pa.G*(e00-e00/3)
        exp2 = exp3 = -2*pa.G*e00/3

        calc = self.get_calc(StressRateD.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')

        self.check(pa.tmp1, exp1, 1e-2)
        self.check(pa.tmp2, exp2, 1e-2)
        self.check(pa.tmp3, exp3, 1e-2)


    def test_stress_rate_d_2(self):
        # TODO: also check the rotation rate effect
        pa = self.pa
        pa.u = self.f
        pa.sigma01 = self.g
        e00 = self.fx
        e01 = self.fy/2
        e02 = self.fz/2
        r01 = self.fy/2
        r02 = self.fz/2

        exp1 = 2*pa.G*(e00-e00/3)
        exp2 = exp3 = -2*pa.G*e00/3

        calc = self.get_calc(StressRateD.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')

        self.check(pa.tmp1, exp1, 1e-2)
        self.check(pa.tmp2, exp2, 1e-2)
        self.check(pa.tmp3, exp3, 1e-2)

    def test_eval_vel_grad(self):
        # TODO: also check the rotation rate effect
        pa = self.pa
        pa.u = self.f
        pa.v = self.g
        pa.w = self.h
        
        exp = numpy.array([[self.fx,self.fy,self.fz],
                           [self.gx,self.gy,self.gz],
                           [self.hx,self.hy,self.hz]])

        calc = self.get_calc(StressRateD.withargs(xsph=False))
        func = calc.funcs[0]
        gV = func.eval_vel_grad_py(self.kernel)
        #print gV
        for dest_pid in self.idx:
            r = gV[:,:,dest_pid]
            e = exp[:,:,dest_pid]
            self.assertTrue(numpy.allclose(r, e, rtol=0.3, atol=1e-2),
                            msg='pid:%d: got: %s\nexp:%s\nerr: %s'%(dest_pid,
                                                   r, e, abs(r-e)/r,
                                                   ))


    def test_stress_rate(self):
        # TODO: also check the rotation rate effect
        pa = self.pa
        pa.u = self.f
        pa.sigma00 = self.h
        pa.sigma01 = self.g
        #pa.sigma11 = -self.h
        e00 = self.fx
        e01 = self.fy/2
        e02 = self.fz/2
        r01 = self.fy/2
        r02 = self.fz/2
        s00 = pa.sigma00
        s01 = pa.sigma01

        exp1 = -s01*self.fz/2
        exp2 = pa.G*self.fz - s00*self.fz/2
        exp3 = pa.G*self.fy - s00*self.fy/2

        calc = self.get_calc(StressRateS.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        
        #print exp1, self.f, self.fz
        self.check(pa.tmp1, exp1, 2e-2)
        self.check(pa.tmp2, exp2, 2e-2)
        self.check(pa.tmp3, exp3, 2e-2)

        
        # check diagonal too
        exp1 = 4*pa.G*self.fx/3 + self.g*self.fy
        exp2 = -2*pa.G*self.fx/3 - self.g*self.fy
        exp3 = -2*pa.G*self.fx/3

        calc = self.get_calc(StressRateD.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        
        #print exp1, self.f, self.fz
        self.check(pa.tmp1, exp1, 1e-2)
        self.check(pa.tmp2, exp2, 1e-2)
        self.check(pa.tmp3, exp3, 1e-2)

    def test_stress_acc(self):
        raise nose.SkipTest('This test fails!')
        # TODO: also check the rotation rate effect
        pa = self.pa
        #pa.u = self.f
        pa.sigma00 = self.h
        pa.sigma01 = self.g
        #pa.sigma11 = -self.h
        # e00 = self.fx
        # e01 = self.fy/2
        # e02 = self.fz/2
        # r01 = self.fy/2
        # r02 = self.fz/2
        s00 = pa.sigma00
        s01 = pa.sigma01

        exp1 = self.hx + self.gy
        exp2 = self.gx
        exp3 = numpy.zeros_like(pa.x)

        calc = self.get_calc(SimpleStressAcceleration.withargs(xsph=False))
        calc.sph('tmp1', 'tmp2', 'tmp3')
        
        #print exp1, self.f, self.fz
        self.check(pa.tmp1, exp1, 2e-2)
        self.check(pa.tmp2, exp2, 2e-2)
        self.check(pa.tmp3, exp3, 2e-2)

 


if __name__ == '__main__':
    nose.main()
