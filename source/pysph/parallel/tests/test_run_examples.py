""" Module to run the example files and report their success/failure results

Add a function to the ExampleTest class corresponding to an example script to
be tested.
This is done till better strategy for parallel testing is implemented
"""

import unittest
from subprocess import Popen, PIPE
from threading import Timer
import os
import sys
import tempfile
import shutil
import numpy

directory = os.path.dirname(os.path.abspath(__file__))

def kill_process(process):
    print 'KILLING PROCESS ON TIMEOUT'
    process.kill()

def run_example_script(filename, args=[], nprocs=2, timeout=20.0, path=None):
    """ run a file python script
    
    Parameters:
    -----------
    filename - filename of python script to run under mpi
    nprocs - (2) number of processes of the script to run (0 => serial non-mpi run)
    timeout - (5) time in seconds to wait for the script to finish running,
        else raise a RuntimeError exception
    path - the path under which the script is located
        Defaults to the location of this file (__file__), not curdir
    
    """
    if path is None:
        path = directory
    path = os.path.join(path, filename)
    if nprocs > 0:
        cmd = ['mpiexec','-n', str(nprocs), sys.executable, path] + args
    else:
        cmd = [sys.executable, path] + args
    print 'running test:', cmd
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout, kill_process, [process])
    timer.start()
    out, err = process.communicate()
    timer.cancel()
    retcode = process.returncode
    if retcode:
        msg = 'test ' + filename + ' failed with returncode ' + str(retcode)
        print out
        print err
        print '#'*80
        print msg
        print '#'*80
        raise RuntimeError, msg
    return retcode, out, err

def get_result(dir1, dir2, prefix, props=['x','y','z']):
    ''' get the results from two runs with output in directories dir1 and dir2 '''
    f1 = [i for i in os.listdir(dir1) if i.startswith(prefix) and i.endswith('.npz')]
    pa_names = set([i.split('_')[-2] for i in f1])
    pa1 = {}
    for f in f1:
        pa = f.split('_')[-2]
        pa1[pa] = {}
        d = numpy.load(os.path.join(dir1,f))
        for prop,val in d.iteritems():
            if numpy.array(val).ndim == 0:
                continue
            if prop in pa1[pa]:
                pa1[pa][prop] = numpy.concatenate([pa1[pa][prop], val])
            else:
                pa1[pa][prop] = val

    f2 = [i for i in os.listdir(dir2) if i.startswith(prefix) and i.endswith('.npz')]
    pa2 = {}
    for f in f2:
        pa = f.split('_')[-2]
        if pa not in pa2:
            pa2[pa] = {}
        d = numpy.load(os.path.join(dir2,f))
        for prop,val in d.iteritems():
            if numpy.array(val).ndim == 0:
                continue
            if prop in pa2[pa]:
                pa2[pa][prop] = numpy.concatenate([pa2[pa][prop], val])
            else:
                pa2[pa][prop] = val
    
    return pa1, pa2

class ExampleTest(unittest.TestCase):
    """ Testcase to run example scripts and compare serial/parallel results """
    def run_example(self, filename, timestep=1e-5, iters=10, nprocs=2,
                    timeout=10, path=None):
        prefix = os.path.splitext(os.path.basename(filename))[0]
        
        try:
            dir1 = tempfile.mkdtemp()
            dir2 = tempfile.mkdtemp()

            args = ['--output=%s'%prefix,
                    '--directory=%s'%dir1,
                    '--time-step=%g'%timestep,
                    '--final-time=%g'%(timestep*(iters+1)),
                    '--freq=%d'%iters]
            run_example_script(filename, args, 0, timeout, path)
            
            args[1] = '--directory=%s'%dir2
            run_example_script(filename, args, nprocs, timeout, path)
            
            pa1, pa2 = get_result(dir1, dir2, prefix)
        finally:
            shutil.rmtree(dir1, True)
            shutil.rmtree(dir2, True)
        
        for pa_name in pa1:
            for prop in pa1[pa_name]:
                p1 = numpy.array(pa1[pa_name][prop])
                if p1.ndim > 0:
                    p1 = numpy.array(sorted(pa1[pa_name][prop]))
                    p2 = numpy.array(sorted(pa2[pa_name][prop]))
                    self.assertTrue(numpy.allclose(p1, p2),
                                    msg='arrays not equal: %r and %r for property'
                                        ' %s of array %s, failing indices: %s'%
                                        (p1, p2, prop, pa_name, numpy.where(p1-p2)))
                else:
                    self.assertAlmostEqual(pa1[pa_name][prop], pa2[pa_name][prop])


    def test_elliptical_drop(self):
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=10, nprocs=2, timeout=10)

if __name__ == "__main__":
    unittest.main()
    
