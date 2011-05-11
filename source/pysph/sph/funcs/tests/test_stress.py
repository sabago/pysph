
import unittest
from numpy import random, linalg, empty, allclose

from pysph.sph.funcs.stress_funcs import py_det, py_get_eigenvalues, py_get_eigenvector

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
    
    def test_eigenvectors(self):
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
        
if __name__ == '__main__':
    unittest.main()
