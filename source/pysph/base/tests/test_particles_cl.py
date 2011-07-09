""" Test the CLParticles class """

from pysph.base.particles import CLParticles
from pysph.base.particle_array import get_particle_array
from pysph.base.locator import OpenCLNeighborLocatorType
from pysph.base.domain_manager import DomainManagerType, LinkedListManager

from pysph.solver.cl_utils import create_some_context, HAS_CL, enqueue_copy

import unittest
import numpy

if HAS_CL:
    import pyopencl as cl
    mf = cl.mem_flags
else:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass

class CLParticlesTestCase(unittest.TestCase):
    """ The CLParticles class is tested. 

    """

    def setUp(self):

        self.np = np = 4

        x = numpy.array([0, 1, 1, 0])
        y = numpy.array([0, 0, 1, 1])
        h = numpy.ones_like(x) * 0.3

        pa = get_particle_array(name="test", cl_precision="single",
                                x=x, y=y, h=h)

        self.particles = particles = CLParticles(
            [pa,], OpenCLNeighborLocatorType.LinkedListSPHNeighborLocator,
            DomainManagerType.LinkedListManager)

        self.ctx = create_some_context()

    def test_constructor(self):

        particles = self.particles

        self.assertEqual( particles.in_parallel, False )
        self.assertEqual( particles.get_cl_precision(), "single" )

    def test_get_domain_manager(self):

        particles = self.particles

        domain_manager = particles.get_domain_manager(self.ctx)

        self.assertEqual( domain_manager.cl_precision, "single" )

        self.assertEqual( domain_manager.with_cl, True )

        cell_size = numpy.float32(0.6)
        self.assertAlmostEqual( domain_manager.cell_size, cell_size, 10 )
        self.assertEqual( domain_manager.ncells, 4 )

        np = 4
        ncells = 4

        head = domain_manager.head['test']
        locks = domain_manager.locks['test']
        next = domain_manager.next['test']
        cellids = domain_manager.cellids['test']
        ix = domain_manager.ix['test']
        iy = domain_manager.iy['test']
        iz = domain_manager.iz['test']

        for i in range(np):
            self.assertAlmostEqual( next[i], -1, 10 )
            self.assertAlmostEqual( cellids[i], 1.0, 10 )
            self.assertAlmostEqual( ix[i], 1.0, 10 )
            self.assertAlmostEqual( iy[i], 1.0, 10 )
            self.assertAlmostEqual( iz[i], 1.0, 10 )

        for i in range(ncells):
            self.assertAlmostEqual( head[i], -1, 10 )
            self.assertAlmostEqual( locks[i], 0, 10 )

    def test_update(self):

        particles = self.particles
        
        particles.setup_cl(self.ctx)    

        pa = self.particles.arrays[0]
        domain_manager = particles.domain_manager

        self.assertEqual( pa.is_dirty, False )
        self.assertEqual( domain_manager.is_dirty, False )

        cellids = domain_manager.cellids['test']
        ix = domain_manager.ix['test']
        iy = domain_manager.iy['test']
        iz = domain_manager.iz['test']

        domain_manager.enqueue_copy()

        self.assertEqual( ix[0], 0 )
        self.assertEqual( ix[1], 1 )
        self.assertEqual( ix[2], 1 )
        self.assertEqual( ix[3], 0 )

        self.assertEqual( iy[0], 0 )
        self.assertEqual( iy[1], 0 )
        self.assertEqual( iy[2], 1 )
        self.assertEqual( iy[3], 1 )

        self.assertEqual( cellids[0], 0 )
        self.assertEqual( cellids[1], 1 )
        self.assertEqual( cellids[2], 3 )
        self.assertEqual( cellids[3], 2 )

    def test_move_particles(self):
        """ Move the particles, set the dirty flag to true and recompute """
        
        particles = self.particles
        pa = particles.arrays[0]
        
        particles.setup_cl(self.ctx)    

        pa = self.particles.arrays[0]
        domain_manager = particles.domain_manager

        q = cl.CommandQueue(self.ctx)

        device_x = pa.get_cl_buffer('x')
        device_y = pa.get_cl_buffer('y')

        xnew = numpy.array([1,0,0,1]).astype(numpy.float32)
        ynew = numpy.array([0,0,1,1]).astype(numpy.float32)

        enqueue_copy(q, src=xnew, dst=device_x)
        enqueue_copy(q, src=ynew, dst=device_y)

        pa.set_dirty(True)

        particles.update()

        self.assertEqual( pa.is_dirty, False )

        cellids = domain_manager.cellids['test']
        ix = domain_manager.ix['test']
        iy = domain_manager.iy['test']
        iz = domain_manager.iz['test']

        domain_manager.enqueue_copy()

        self.assertEqual( ix[0], 1 )
        self.assertEqual( ix[1], 0 )
        self.assertEqual( ix[2], 0 )
        self.assertEqual( ix[3], 1 )

        self.assertEqual( iy[0], 0 )
        self.assertEqual( iy[1], 0 )
        self.assertEqual( iy[2], 1 )
        self.assertEqual( iy[3], 1 )

        self.assertEqual( cellids[0], 1 )
        self.assertEqual( cellids[1], 0 )
        self.assertEqual( cellids[2], 2 )
        self.assertEqual( cellids[3], 3 )


if __name__ == '__main__':
    unittest.main()
