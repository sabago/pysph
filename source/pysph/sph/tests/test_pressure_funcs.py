""" Tests for the pressure force functions """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

import numpy
import unittest

from function_test_template import FunctionTestCase

NSquareLocator = base.NeighborLocatorType.NSquareNeighborLocator

class PressureGradientTestCase(FunctionTestCase):

    def setup(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square. The pressure gradient term to be
        tested is

        ..math::

                \frac{\nablaP}{\rho}_i = \sum_{j=1}^{4}
                -m_j(\frac{Pa}{\rho_a^2} + \frac{Pb}{\rho_b^2})\nabla W_{ab}

        The mass of each particle is 1

        """

        self.func = sph.SPHPressureGradient.withargs().get_func(self.pa,
                                                                self.pa)

        #grad_func = sph.SPHPressureGradient.withargs()


        # self.grad_func = grad_func.get_func(pa,pa)
        # self.mom_func = mom_func.get_func(pa,pa)
        
        # self.grad_func.kernel = base.CubicSplineKernel(dim=2)
        # self.grad_func.nbr_locator = \
        #                       base.Particles.get_neighbor_particle_locator(pa,
        #                                                                    pa)

        # self.mom_func.kernel = base.CubicSplineKernel(dim=2)
        # self.mom_func.nbr_locator = \
        #                      base.Particles.get_neighbor_particle_locator(pa,
        #                                                                   pa)

        # self.setup_cl()

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        forces = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            force = base.Point()
            xi, yi, zi = x[i], y[i], z[i]

            ri = base.Point(xi,yi,zi)

            Pi, rhoi = p[i], rho[i]
            hi = h[i]

            for j in range(self.np):

                grad = base.Point()
                xj, yj, zj = x[j], y[j], z[j]
                Pj, rhoj = p[j], rho[j]
                hj, mj = m[j], h[j]

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                tmp = -mj * ( Pi/(rhoi*rhoi) + Pj/(rhoj*rhoj) )
                kernel.py_gradient(ri, rj, havg, grad)

                force.x += tmp*grad.x
                force.y += tmp*grad.y
                force.z += tmp*grad.z

            forces.append(force)

        return forces

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=6)


class MomentumEquationTestCase(FunctionTestCase):

    def setup(self):
        self.func = sph.MomentumEquation.withargs(
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
            xi, yi, zi = x[i], y[i], z[i]
            ui, vi, wi = u[i], v[i], w[i]

            ri = base.Point(xi,yi,zi)
            Va = base.Point(ui,vi,wi)

            Pi, rhoi = p[i], rho[i]
            hi = h[i]

            for j in range(self.np):

                grad = base.Point()
                xj, yj, zj = x[j], y[j], z[j]
                Pj, rhoj = p[j], rho[j]
                hj, mj = h[j], m[j]

                uj, vj, wj = u[j], v[j], w[j]
                Vb = base.Point(uj,vj,wj)

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                tmp = Pi/(rhoi*rhoi) + Pj/(rhoj*rhoj)
                kernel.py_gradient(ri, rj, havg, grad)

                vab = Va-Vb
                rab = ri-rj

                dot = vab.dot(rab)
                piab = 0.0

                if dot < 0.0:
                    alpha = 1.0
                    beta = 1.0
                    gamma = 1.4
                    eta = 0.1

                    cab = 0.5 * (cs[i] + cs[j])

                    rhoab = 0.5 * (rhoi + rhoj)
                    muab = havg * dot

                    muab /= ( rab.norm() + eta*eta*havg*havg )

                    piab = -alpha*cab*muab + beta*muab*muab
                    piab /= rhoab

                tmp += piab
                tmp *= -mj
                    
                force.x += tmp*grad.x
                force.y += tmp*grad.y
                force.z += tmp*grad.z

            forces.append(force)

        return forces

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=6)
                

if __name__ == '__main__':
    unittest.main()            
