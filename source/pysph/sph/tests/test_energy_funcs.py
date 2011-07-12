""" Tests for the energy force functions """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

import numpy
import unittest

from function_test_template import FunctionTestCase

NSquareLocator = base.NeighborLocatorType.NSquareNeighborLocator

class EnergyEquationNoViscTestCase(FunctionTestCase):

    def setup(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square.

        The function tested is

        ..math::

        \frac{DU_a}{Dt} = \frac{1}{2}\sum_{b=1}^{N}m_b\left[
        \left(\frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2}\right)\,(v_a -
        v_b)\right]\,\nabla_a \cdot W_{ab}

        The mass of each particle is 1

        """
        

        self.func = sph.EnergyEquationNoVisc.withargs().get_func(self.pa,
                                                                 self.pa)
        
    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        forces = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')
        u,v,w = pa.get('u','v','w')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            force = base.Point()
            xa, ya, za = x[i], y[i], z[i]
            ua, va, wa = u[i], v[i], w[i]

            ra = base.Point(xa,ya,za)
            Va = base.Point(ua,va,wa) 

            Pa, rhoa = p[i], rho[i]
            ha = h[i]

            for j in range(self.np):

                grad = base.Point()
                xb, yb, zb = x[j], y[j], z[j]
                ub, vb, wb = u[j], v[j], w[j]

                Pb, rhob = p[j], rho[j]
                hb, mb = m[j], h[j]

                havg = 0.5 * (ha + hb)

                rb = base.Point(xb, yb, zb)
                Vb = base.Point(ub, vb, wb)
        
                tmp = 0.5*mb * ( Pa/(rhoa*rhoa) + Pb/(rhob*rhob) )
                kernel.py_gradient(ra, rb, havg, grad)

                force.x += tmp * grad.dot(Va-Vb)

            forces.append(force)

        return forces

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=6)

class EnergyEquationTestCase(FunctionTestCase):

    def setup(self):
        self.func = sph.EnergyEquation.withargs(
            alpha=1.0, beta=1.0, gamma=1.4, eta=0.1).get_func(self.pa,
                                                              self.pa)
    
    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        forces = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')
        u,v,w,cs = pa.get('u','v','w','cs')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            force = base.Point()
            xa, ya, za = x[i], y[i], z[i]
            ua, va, wa = u[i], v[i], w[i]

            ra = base.Point(xa,ya,za)
            Va = base.Point(ua,va,wa)

            Pa, rhoa = p[i], rho[i]
            ha = h[i]

            for j in range(self.np):

                grad = base.Point()
                xb, yb, zb = x[j], y[j], z[j]
                Pb, rhob = p[j], rho[j]
                hb, mb = h[j], m[j]

                ub, vb, wb = u[j], v[j], w[j]
                Vb = base.Point(ub,vb,wb)

                havg = 0.5 * (hb + ha)

                rb = base.Point(xb, yb, zb)
        
                tmp = Pa/(rhoa*rhoa) + Pb/(rhob*rhob)
                kernel.py_gradient(ra, rb, havg, grad)

                vab = Va-Vb
                rab = ra-rb

                dot = vab.dot(rab)
                piab = 0.0

                if dot < 0.0:
                    alpha = 1.0
                    beta = 1.0
                    gamma = 1.4
                    eta = 0.1

                    cab = 0.5 * (cs[i] + cs[j])

                    rhoab = 0.5 * (rhoa + rhob)
                    muab = havg * dot

                    muab /= ( rab.norm() + eta*eta*havg*havg )

                    piab = -alpha*cab*muab + beta*muab*muab
                    piab /= rhoab

                tmp += piab
                tmp *= 0.5*mb

                force.x += tmp * ( vab.dot(grad) )

            forces.append(force)

        return forces

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=6)

if __name__ == '__main__':
    unittest.main()            
