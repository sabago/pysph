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

from pysph.solver.utils import load

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

def _concatenate_output(arrays, nprocs):
    if nprocs <= 0:
        return 0

    array_names = arrays[0].keys()

    first_processors_arrays = arrays[0]
    
    if nprocs > 1:
        ret = {}
        for array_name in array_names:
            first_array = first_processors_arrays[array_name]
            for rank in range(1,nprocs):
                other_processors_arrays = arrays[rank]
                other_array = other_processors_arrays[array_name]

                # append the other array to the first array
                first_array.append_parray(other_array)

                # remove the non local particles
                first_array.remove_tagged_particles(1)
                
            ret[array_name] = first_array

    else:
        ret = arrays[0]

    return ret

def _get_result(directory, nprocs, prefix):
    """ Return the results from a PySPH run.

    The return value is a dictionary keyed on particle array names and
    the particle array as value. Results from multiple runs are
    concatenated to a single particle array.

    Parameters:
    -----------

    directory : str
        The directory for the results

    nprocs : int
        The number of processors the run consisted of

    prefix : str
        The file name prefix to load the result

    """

    _file = [i.rsplit('_',1)[1][:-4] for i in os.listdir(directory) if i.startswith(prefix) and i.endswith('.npz')][-1]

    arrays = {}

    for rank in range(nprocs):
        fname = os.path.join(directory, prefix+'_'+str(rank)+'_'+str(_file)+'.npz')

        arrays[rank] = load(fname)["arrays"]

    return _concatenate_output(arrays, nprocs)

class ExampleTestCase(unittest.TestCase):
    """ A script to run an example in serial and parallel and compare results.

    To test an example in parallel, subclass from ExampleTest and
    write a test function like so:

    def test_elliptical_drop(self):
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=100, nprocs=2, timeout=60)
    
    """
    def run_example(self, filename, timestep=1e-5, iters=10, nprocs=2,
                    timeout=10, path=None):
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
                    '--time-step=%g'%timestep,
                    '--final-time=%g'%(timestep*(iters+1)),
                    '--freq=%d'%iters]

            # run the example script in serial
            _run_example_script(filename, args, 0, timeout, path)

            # run the example script in parallel
            args[1] = '--directory=%s'%dir2
            _run_example_script(filename, args, nprocs, timeout, path)

            # get the serial and parallel results
            serial_result = _get_result(directory=dir1, nprocs=1, prefix=prefix)
            parallel_result = _get_result(directory=dir2, nprocs=nprocs,
                                          prefix=prefix)
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

    def _test_elliptical_drop(self, nprocs, iter, timeout):
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=iter,
                         nprocs=nprocs, timeout=timeout)

    def test_elliptical_drop_2(self):
        """Test with 2 processors """
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=100, nprocs=2, timeout=60)

    def test_elliptical_drop_4(self):
        """Test with 4 processors """
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=100, nprocs=4, timeout=240)

if __name__ == "__main__":
    unittest.main()
    
