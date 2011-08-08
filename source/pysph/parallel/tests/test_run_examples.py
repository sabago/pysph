""" Module to run the example files and report their success/failure results

Add a function to the ExampleTest class corresponding to an example script to
be tested.
This is done till better strategy for parallel testing is implemented
"""

try:
    import mpi4py.MPI as mpi
except ImportError:
    import nose.plugins.skip as skip
    reason = "mpi4py not installed"
    raise skip.SkipTest(reason)

import unittest
from subprocess import Popen, PIPE
from threading import Timer
import os
import sys
import tempfile
import shutil
import numpy

from nose.plugins.attrib import attr

import pysph.solver.utils as utils

directory = os.path.dirname(os.path.abspath(__file__))

def kill_process(process):
    print 'KILLING PROCESS ON TIMEOUT'
    process.kill()

def _run_example_script(filename, args=[], nprocs=2, timeout=20.0, path=None):
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

class ExampleTestCase(unittest.TestCase):
    """ A script to run an example in serial and parallel and compare results.

    To test an example in parallel, subclass from ExampleTest and
    write a test function like so:

    def test_elliptical_drop(self):
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=100, nprocs=2, timeout=60)
    
    """
    def run_example(self, filename, timestep=1e-5, iters=10, nprocs=2,
                    timeout=10, parallel_mode="simple",path=None):
        """Run an example and compare the results in serial and parallel.

        Parameters:
        -----------

        filename : str
            The name of the file to run

        timestep : double
            The time step argument to pass to the example script

        iters : double
            The number of iteations to evolve the example

        nprocs : int
            Number of processors to use for the example.

        timeout : int
            Time in seconds to wait for execution before an error is raised.

        path : Not used

        """
        prefix = os.path.splitext(os.path.basename(filename))[0]
        
        try:
            # dir1 is for the serial run
            dir1 = tempfile.mkdtemp()

            # dir2 is for the parallel run 
            dir2 = tempfile.mkdtemp()

            args = ['--output=%s'%prefix,
                    '--directory=%s'%dir1,
                    '--timestep=%g'%timestep,
                    '--final-time=%g'%(timestep*(iters+1)),
                    '--freq=%d'%iters,
                    '--parallel-mode=%s'%parallel_mode]

            # run the example script in serial
            _run_example_script(filename, args, 0, timeout, path)

            # run the example script in parallel
            args[1] = '--directory=%s'%dir2
            _run_example_script(filename, args, nprocs, timeout, path)

            # get the serial and parallel results
            serial_result = utils.load_and_concatenate(
                directory=dir1, nprocs=1, prefix=prefix)["arrays"]

            parallel_result = utils.load_and_concatenate(
                directory=dir2, nprocs=1, prefix=prefix)["arrays"]
                                                       
        finally:
            shutil.rmtree(dir1, True)
            shutil.rmtree(dir2, True)

        # test
        self._test(serial_result, parallel_result)

    def _test(self, serial_result, parallel_result):

        # make sure the array names are the same
        serial_arrays = serial_result.keys()
        parallel_arrays = parallel_result.keys()

        self.assertTrue( serial_arrays == parallel_arrays )

        arrays = serial_arrays
        
        # test the results.
        for array in arrays:
            
            x_serial, y_serial, z_serial  = serial_result[array].get("x","y","z")
            x_par, y_par, z_par = parallel_result[array].get("x", "y", "z")

            np = len(x_serial)
            self.assertTrue( len(x_par) == np )

            idx = parallel_result[array].get("idx")
            for i in range(np):
                self.assertAlmostEqual( x_serial[idx[i]], x_par[i], 10 )
                self.assertAlmostEqual( y_serial[idx[i]], y_par[i], 10 )
                self.assertAlmostEqual( z_serial[idx[i]], z_par[i], 10 )


class EllipticalDropTestCase(ExampleTestCase):

    def _test_elliptical_drop(self, nprocs, iter, timeout, parallel_mode):
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=iter,
                         nprocs=nprocs, timeout=timeout,
                         parallel_mode=parallel_mode)

    @attr(slow=True, parallel=True)
    def test_elliptical_drop_simple_2(self):
        """Test with 2 processors """
        self._test_elliptical_drop(nprocs=2, iter=100, timeout=120,
                                   parallel_mode="simple")

    @attr(slow=True, parallel=True)
    def test_elliptical_drop_simple_4(self):
        """Test with 4 processors """
        self._test_elliptical_drop(nprocs=4, iter=100, timeout=360,
                                   parallel_mode="simple")
    @attr(slow=True, parallel=True)
    def test_elliptical_drop_block_2(self):
        self._test_elliptical_drop(nprocs=2, iter=100, timeout=240,
                                   parallel_mode="block")

    @attr(slow=True, parallel=True)
    def test_elliptical_drop_block_4(self):
        self._test_elliptical_drop(nprocs=4, iter=100, timeout=360,
                                   parallel_mode="block")

if __name__ == "__main__":
    unittest.main()
    
