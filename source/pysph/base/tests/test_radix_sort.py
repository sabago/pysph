"""Tests for the CPU version of the OpenCL Radix Sort"""

import numpy.random as random
import numpy

import unittest

# PySPH imports
import pysph.base.api as base
from pysph.solver.api import get_real, HAS_CL

if not HAS_CL:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass

import pysph.solver.cl_utils as clu

class AMDRadixSortTestCase(unittest.TestCase):
    """Test for the  AMD's OpenCL radix sort implementation.

    Currently we test the keys only approach which 
    sorts an input set of keys which are unsigned
    integers (32 bit)
    
    """

    def setUp(self):
        """General setup to test the Radix sort procedure"""

        # limits for the keys: We assume the keys are 32 bit
        # unsigned integers so the maximum representable
        # value is (2*32) - 1. In addition, we reserve this last
        # value for padding. So all padded elements have this value.
        uint_low = 0
        uint_high = (1<<32) - 2

        # generate the random points
        self.np = np = 30000
        keys = random.random_integers( low=uint_low, high=uint_high,
                                       size=(np) ).astype(numpy.uint32)
    
        self.keys = keys

        # store the reference sorted keys for comparison
        self.sortedkeys = numpy.sort(keys, kind="mergesort")
        
        # instantiate the RadixSortManager
        self.rsort = base.AMDRadixSort(radix=8)
        self.rsort.initialize(keys, values=None)
        
    def test_constructor(self):
        """Test the basic construction for AMDRaidxSort"""

        rsort = self.rsort

        # test the radix, radices and groupsize constants
        self.assertEqual( rsort.radix, 8 )
        self.assertEqual( rsort.radices, 256 )
        self.assertEqual( rsort.group_size, 64 )

        # test the nelements and num groups
        self.assertEqual( rsort.nelements, 32768 )
        self.assertEqual( rsort.num_groups, 2 )

        # test the padding of the keys
        keys = rsort._keys
        val = clu.uint32mask()
        for i in range( rsort.n, rsort.nelements ):
            self.assertEqual( keys[i], val )

    def test_sort(self):
        """Test the sorting routine"""
        rsort = self.rsort

        # do the OpenCL setup
        rsort._setup_cl()

        # now sort the data
        rsort.sort()

        sol = rsort.keys
        ref = self.sortedkeys

        # test the sort
        for i in range(rsort.n):
            self.assertEqual( sol[i], ref[i] )
        

if __name__ == "__main__":
    unittest.main()
