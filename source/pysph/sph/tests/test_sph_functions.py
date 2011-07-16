""" test sanity of various functions """

import unittest
import time
import numpy

from pysph.base import kernels
from pysph.base.carray import DoubleArray
from pysph.base.particle_array import ParticleArray
from pysph.base.api import get_particle_array, Particles
from pysph.sph.sph_calc import SPHCalc
from pysph.sph.sph_func import get_all_funcs
from pysph.sph.funcs.stress_funcs import get_G, get_K, get_nu

import nose

funcs = get_all_funcs()

Ns = [100]#, 100000]

class TestSPHFuncs(unittest.TestCase):
    pass


# function names have 't' instead of 'test' otherwise nose test collector
# assumes them to be test functions
def create_t_func(func_getter):
    """ create and return test functions for sph_funcs """
    cls = func_getter.get_func_class()
    
    def t(self):
        ret = {}
        da = DoubleArray()
        pa = ParticleArray()
        kernel = kernels.CubicSplineKernel(3)
        get_time = time.time
        for N in Ns:
            x = numpy.arange(N)
            z = y = numpy.zeros(N)
            mu = m = rho = numpy.ones(N)
            h = 2*m
            da = DoubleArray(N)
            da2 = DoubleArray(N)
            da.set_data(z)
            da2.set_data(z)
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
                                    rkpm_dbeta2dy=z,
                                    S_00=z, S_01=z, S_02=z,
                                    S_10=z, S_11=z, S_12=z,
                                    S_20=z, S_21=z, S_22=z,
                                    R_00=z, R_01=z, R_02=z,
                                    R_11=z, R_12=z, R_22=z,
                                    v_00=z, v_01=z, v_02=z,
                                    v_10=z, v_11=z, v_12=z,
                                    v_20=z, v_21=z, v_22=z,
                                    name="pa")
            pa.constants['E'] = 1e9
            pa.constants['nu'] = 0.3
            pa.constants['G'] = pa.constants['E']/(2.0*1+pa.constants['nu'])
            pa.constants['K'] = get_K(pa.constants['G'], pa.constants['nu'])
            pa.constants['rho0'] = 1.0
            pa.constants['dr0'] = 1.0
            pa.constants['c_s'] = numpy.sqrt(pa.constants['K']/pa.constants['rho0'])

            pb = get_particle_array(x=x+0.1**0.5, y=y, z=z, h=h, mu=mu,
                                    rho=rho, m=m, tmp=z,
                                    tx=m, ty=z, tz=z,
                                    nx=z, ny=m, nz=z, u=z, v=z, w=z,
                                    ubar=z, vbar=z, wbar=z,
                                    q=m, div=z, rhop=z, e_t=m*0.1,
                                    rkpm_beta1=z,
                                    rkpm_beta2=z,
                                    rkpm_alpha=z,
                                    rkpm_dbeta1dx=z,
                                    rkpm_dbeta1dy=z,
                                    rkpm_dbeta2dx=z,
                                    rkpm_dbeta2dy=z,
                                    S_00=z, S_01=z, S_02=z,
                                    S_10=z, S_11=z, S_12=z,
                                    S_20=z, S_21=z, S_22=z,
                                    R_00=z, R_01=z, R_02=z,
                                    R_11=z, R_12=z, R_22=z,
                                    v_00=z, v_01=z, v_02=z,
                                    v_10=z, v_11=z, v_12=z,
                                    v_20=z, v_21=z, v_22=z,
                                    name="pb")
            pb.constants['E'] = 1e9
            pb.constants['nu'] = 0.3
            pb.constants['G'] = pb.constants['E']/(2.0*1+pb.constants['nu'])
            pb.constants['K'] = get_K(pb.constants['G'], pb.constants['nu'])
            pb.constants['rho0'] = 1.0
            pb.constants['dr0'] = 1.0
            pb.constants['c_s'] = numpy.sqrt(pb.constants['K']/pb.constants['rho0'])
            
            particles = Particles(arrays=[pa, pb])
            
            func = func_getter.get_func(pa, pb)
            calc = SPHCalc(particles, [pa], pb, kernel, [func], ['_tmpx'])
            print cls.__name__
            t = get_time()
            if cls.__name__.startswith("Hookes"):
                calc.tensor_sph()
            else:
                calc.sph()
            t = get_time() - t
            
            nam = '%s'%(cls.__name__)
            ret[nam +' /%d'%(N)] = t/N
        return ret

    t.__name__ = 'test_sph_func__%s'%(cls.__name__)
    t.__doc__ = 'run sanity check for calc: %s'%(cls.__name__)
    
    return t


def gen_ts():
    """ generate test functions and attach them to test classes """
    raise nose.SkipTest('Disabling!')
    for i, func in enumerate(funcs.values()):
        t_method = create_t_func(func)
        t_method.__name__ = t_method.__name__ + '_%d'%(i)
        setattr(TestSPHFuncs, t_method.__name__, t_method)

# generate the test functions
gen_ts()
    
if __name__ == "__main__":
    unittest.main()
