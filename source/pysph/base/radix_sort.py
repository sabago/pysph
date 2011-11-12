"""Module to implement the RadixSortManagers for CPU and GPU"""

# OpenCL conditional imports
import pysph.solver.cl_utils as clu

if clu.HAS_CL:
    import pyopencl as cl
    mf = cl.mem_flags

import numpy

class AMDRadixSort:
    """AMD's OPenCL implementation of the radix sort.

    The C++ code for the implementation can be found in
    the samples drectory of the AMD-APP-SDK (2.5)

    This implementation assumes that each thread works
    on 256 elements. This is also the minimum number of
    elements required. The keys are assumed to be 32 bit
    unsigned bits and we sort them 8 bits at a time. As
    a result, the number of histogram bins/buckets for
    this implementation is also 256.    

    """

    def __init__(self, keys, values=None, radix=8):
        """Constructor.

        Parameters:
        ------------

        keys : array (uint32)
            The keys used to sort the data

        values : array
            The values to be sorted based on the keys.

        radix : int (8)
            The number of bits per pass of the radix sort.

        """
        
        # store the keys and values
        self.keys = keys
        self.values = values

        # number of elements
        self.n = len( keys )

        # the following variables are analogous to the AMD's
        # variables
        self.radix = radix        # number of bits at a time for each pass
        self.radices = (1<<radix) # num of elements handled by each work-item

        # the group size could be changed to any convenient size
        self.group_size = 64

        self._setup()

    def sort(self):
        """The main sorting routine"""

        ctx = self.context
        q = self.queue

        radices = self.radices
        ngroups = self.num_groups
        groupsize = self.group_size

        histograms = self.histograms

        for _bits in range(0, 32, 8):

            bits = numpy.int32(_bits)
            
            # compute the histogram on the device
            self._histogram( bits )

            # Scan the histogram on the host
            _sum = 0
            for i in range(radices):
                for j in range(ngroups):
                    for k in range(groupsize):
                        index = j * groupsize * radices + k * radices + i
                        value = histograms[index]

                        histograms[index] = _sum
                        _sum += value

            # permute the data on the device
            self._permute( bits )

            # current output becomes input for the next pass
            self._keys[:] = self.sortedkeys[:]

        # read only the original un-padded data into keys
        self.keys[:] = self._keys[:self.n]

    def _histogram(self, bits):
        """Launch the histogram kernel"""

        ctx = self.context
        q = self.queue

        # global/local sizes
        global_sizes = (self.nelements/self.radices,)
        local_sizes = (self.group_size,)

        # copy the unsorted data to the device 
        # the unsorted data is in _keys and dkeys
        clu.enqueue_copy(q, src=self._keys, dst=self.dkeys)

        # allocate the local memory for the histogram kernel
        local_mem_size = self.group_size * self.radices * 2
        local_mem = cl.LocalMemory(size=local_mem_size)

        # enqueue the kernel for execution
        self.program.histogram(q, global_sizes, local_sizes,
                               self.dkeys, self.dhistograms,
                               bits, local_mem).wait()

        # read the result to the host buffer
        clu.enqueue_copy(q, src=self.dhistograms, dst=self.histograms)

    def _permute(self, bits):
        """Launch the permute kernel"""

        ctx = self.context
        q = self.queue

        # copy the scanned histograms to the device
        clu.enqueue_copy(q, src=self.histograms,
                         dst=self.dscanedhistograms)

        # global and local sizes
        global_sizes = (self.nelements/self.radices,)
        local_sizes = (self.group_size,)

        # allocate local memory for the permute kernel launch
        local_mem_size = self.group_size * self.radices * 2
        local_mem = cl.LocalMemory(size=local_mem_size)

        # enqueue the kernel for execution
        self.program.permute(q, global_sizes, local_sizes,
                             self.dkeys, self.dscanedhistograms,
                             bits, local_mem, self.dsortedkeys).wait()

        # read sorted results back to the host
        clu.enqueue_copy(q, src=self.dsortedkeys,
                         dst=self.sortedkeys)

    def _setup(self):
        """Prepare the data for the algorithm

        The implementation requires the input array to have
        a length equal to a power of 2. We test for this
        condition and pad the keys with the special mask
        value (1<<32 - 1) which has a bit pattern of all 1's

        This particular padding and the ordered nature of
        the radix sort results in these padded dummy keys
        going to the end so we can simpy ignore them.

        """

        # check the length of the input arrays
        if not clu.ispowerof2(self.n):
            n = clu.round_up(self.n)
            pad = numpy.ones(n - self.n, numpy.int32) * clu.uint32mask()

            # _keys is the padded array we use internally
            self._keys = numpy.concatenate((self.keys, pad)).astype(numpy.uint32)

            # TODO : PAD THE VALUES AS WELL
            
        else:
            self._keys = self.keys
            self._values = self.values

        # now store the number of elements and num work groups
        self.nelements = len(self._keys)
        self.num_groups = self.nelements/(self.group_size * self.radices)

    def _setup_cl(self, context=None):
        """ OpenCL setup. """

        if context is None:
            self.context = context = clu.create_some_context()
        else:
            self.context = context
        
        self.queue = queue = cl.CommandQueue(context)

        # allocate device memory
        self._allocate_memory()

        # create the program
        self._create_program()
        
    def _allocate_memory(self):
        """Allocate OpenCL work buffers."""

        ctx = self.context

        # first allocate the keys and values on the device
        self.dkeys = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR,
                               hostbuf=self._keys)

        # Output from the histogram kernel
        # each thread will write it's histogram/count
        # for the 256 elements it's checking for. Thus,
        # the size of this buffer should be:
        # numgroups * local_size * num_elements_per work item
        size = self.group_size * self.radices * self.num_groups
        self.histograms = numpy.ones(size, numpy.uint32)
        self.dhistograms = cl.Buffer(ctx, mf.READ_WRITE, size=size*4)

        # Input for the permute kernel.
        # For this kernel, the output for the histogram kernel is
        # scanned and used as input to the permute kernel. Thus,
        # the size requirement is the same.
        self.dscanedhistograms = cl.Buffer(ctx, mf.READ_WRITE, size=size*4)
        
        # the final output or the sorted output.
        # This should obviously be of size num_elements
        self.sortedkeys = numpy.ones(self.nelements, numpy.uint32)
        self.dsortedkeys = cl.Buffer(ctx, mf.READ_WRITE, size=self.nelements*4)

    def _create_program(self):
        """Read the OpenCL kernel file and build"""
        src_file = clu.get_pysph_root() + '/base/RadixSort_Kernels.cl'
        src = open(src_file).read()
        self.program = cl.Program(self.context, src).build()
        
        
