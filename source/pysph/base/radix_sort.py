"""Module to implement the RadixSortManagers for CPU and GPU"""

# OpenCL conditional imports
import pysph.solver.cl_utils as clu

if clu.HAS_CL:
    import pyopencl as cl
    mf = cl.mem_flags

import numpy
from scanClass import Scan

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

    Since we expect this radix sort routine to be used to sort
    particle cell indices as keys and particle indices as values,
    the values are also assumed to be unsigned integers.
    
    """

    def __init__(self, radix=8):
        """Constructor.

        Parameters:
        ------------

        radix : int (8)
            The number of bits per pass of the radix sort.

        """
        
        # the following variables are analogous to the AMD's
        # variables
        self.radix = radix        # number of bits at a time for each pass
        self.radices = (1<<radix) # num of elements handled by each work-item

        # the group size could be changed to any convenient size
        self.group_size = 64

    def initialize(self, keys, values=None, context=None):
        """Initialize the radix sort manager"""
        # store the keys and values
        self.keys = keys

        # a keys only sort treats the values as keys
        if values is None:
            self.values = keys.copy()
        else:
            nvalues = len(values); nkeys = len(keys)
            if not nvalues == nkeys:
                raise RuntimeError( "len(keys) %d != len(values) %d"%(nkeys,
                                                                      nvalues) )
            self.values = values

        # number of elements
        self.n = len( keys )

        # pad etc
        self._setup()

        # OpenCL setup
        self._setup_cl(context)

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
            self._values[:] = self.sortedvalues[:]

        # read only the original un-padded data into keys
        self.keys[:] = self._keys[:self.n]
        self.values[:] = self._values[:self.n]

    def _histogram(self, bits):
        """Launch the histogram kernel

        Each thread will load it's work region (256 values) into
        shared memory and will compute the histogram/frequency of
        occurance of each element. Remember that the implementation
        assumes that we sort the 32 bit keys and values 8 bits at a
        time and as such the histogram bins/buckets for each thread
        are also 256.

        We first copy the currenty unsorted data to the device before
        calculating local memory size and then launching the kernel.

        After the kernel launch, we read the computed thread
        histograms to the host, where these will be scanned.

        """

        ctx = self.context
        q = self.queue

        # global/local sizes
        global_sizes = (self.nelements/self.radices,)
        local_sizes = (self.group_size,)

        # copy the unsorted data to the device 
        # the unsorted data is in _keys and dkeys
        #clu.enqueue_copy(q, src=self._keys, dst=self.dkeys)

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
        """Launch the permute kernel

        Using the host-scanned thread histograms, this kernel shuffles
        the array values in the keys and values to perform the actual
        sort.

        We first copy the scanned histograms to the device, compute
        local mem size and then launch the kernel. After the kernel
        launch, the sorted keys and values are read back to the host
        for the next pass.

        """

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
                             self.dkeys, self.dvalues,
                             self.dscanedhistograms,
                             bits, local_mem,
                             self.dsortedkeys, self.dsortedvalues).wait()

        # read sorted results back to the host
        clu.enqueue_copy(q, src=self.dsortedkeys, dst=self.sortedkeys)
        clu.enqueue_copy(q, src=self.dsortedvalues, dst=self.sortedvalues)
        
        clu.enqueue_copy(q, src=self.dsortedkeys, dst=self.dkeys)
        clu.enqueue_copy(q, src=self.dsortedvalues, dst=self.dvalues)

        
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

            # _keys and _values are the  padded arrays we use internally
            self._keys = numpy.concatenate( (self.keys,
                                             pad) ).astype(numpy.uint32)

            self._values = numpy.concatenate( (self.values,
                                               pad) ).astype(numpy.uint32)
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
        # these serve as the unsorted keys and values on the device
        self.dkeys = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR,
                               hostbuf=self._keys)

        self.dvalues = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR,
                                 hostbuf=self._values)

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

        self.sortedvalues = numpy.ones(self.nelements, numpy.uint32)
        self.dsortedvalues = cl.Buffer(ctx, mf.READ_WRITE,
                                       size=self.nelements*4)

    def _create_program(self):
        """Read the OpenCL kernel file and build"""
        src_file = clu.get_pysph_root() + '/base/RadixSort_Kernels.cl'
        src = open(src_file).read()
        self.program = cl.Program(self.context, src).build()
        
    def _sort_cpu(self, keys, values=None):
        """Perform a reference radix sort for verification on the CPU

        The reference implemetation is analogous to the AMD-APP-SDK's
        reference host implementation.

        """
        if values is None:
            values = keys.copy()

        n = len(keys)

        sortedkeys = numpy.ones(n, numpy.uint32)
        sortedvalues = numpy.ones(n, numpy.uint32)

        mask = self.radices - 1

        # allocate the histogram buffer. This is simply a buffer of
        # length RADICES (256)
        histograms = numpy.zeros(self.radices, numpy.uint32)

        # Sort the data
        for bits in range(0, 32, self.radix):

            # initialize the histograms to 0
            histograms[:] = 0

            # calculate histograms for all elements
            for i in range(n):
                element = keys[i]
                val = (element >> bits) & mask
                histograms[val] += 1

            # scan the histograms (exclusive)
            _sum = 0.0
            for i in range(self.radices):
                val = histograms[i]
                histograms[i] = _sum
                _sum += val

            # permute the keys and values
            for i in range(n):
                element = keys[i]
                val = ( element >> bits ) & mask
                index = histograms[val]
                
                sortedkeys[index] = keys[i]
                sortedvalues[index] = values[i]
                
                histograms[val] = index + 1

            # swap the buffers for the next pass
            keys[:] = sortedkeys[:]
            values[:] = sortedvalues[:]

class NvidiaRadixSort:
    """
    LICENSE
    """

    def __init__(self, radix=8):
        """Constructor.

        Parameters:
        ------------

        radix : int (8)
            The number of bits per pass of the radix sort.

        """
        
        # the following variables are analogous to the AMD's
        # variables
        self.radix = radix        # number of bits at a time for each pass
        self.radices = (1<<radix) # num of elements handled by each work-item

        # the group size could be changed to any convenient size
        self.group_size = 64


    def initialize(self, keys, values=None, context=None):
        """Initialize the radix sort manager"""
        # store the keys and values
        self.keys = keys

        # a keys only sort treats the values as keys
        if values is None:
            self.values = keys.copy()
        else:
            nvalues = len(values); nkeys = len(keys)
            if not nvalues == nkeys:
                raise RuntimeError( "len(keys) %d != len(values) %d"%(nkeys,
                                                                      nvalues) )
            self.values = values

        # number of elements
        self.n = len( keys )

        # pad etc
        self._setup()

        # OpenCL setup
        self._setup_cl(context)

    def sort(self):
        keyBits = self.keyBits
        self.radixSortKeysOnly(keyBits)

        clu.enqueue_copy(self.queue, src=self.dkeys, dst=self.sortedkeys)
        clu.enqueue_copy(self.queue, src=self.dvalues, dst=self.sortedvalues)

        self.keys[:] = self.sortedkeys[:self.n]
        self.values[:] = self.sortedvalues[:self.n]

    def radixSortKeysOnly(self, keyBits):
        i = numpy.uint32(0)
        bitStep = self.bitStep
        
        while (keyBits > i*bitStep):
            self.radixSortStepKeysOnly(bitStep, i*bitStep)
            i+=numpy.uint32(1)

    def radixSortStepKeysOnly(self, nbits, startbit):
        nelements = self.nelements
        
        # create scan object
        scan = Scan(self.context,
                    self.queue,
                    nelements)

        # 4 step algo
        ctaSize = self.ctaSize
        
        # STEP I {radixSortBlocksKeysOnlyOCL}
        totalBlocks = numpy.uint32(nelements/4/ctaSize)
        globalWorkSize = (numpy.int(ctaSize*totalBlocks),)
        localWorkSize = (numpy.int(ctaSize),)

        # create Local Memory
        self.local1 = cl.LocalMemory(size=numpy.int(4 * ctaSize * self.size_uint))

        self.program.radixSortBlocksKeysOnly(self.queue, globalWorkSize, localWorkSize,
                                             self.dkeys,
                                             self.dsortedkeys,
                                             self.dvalues, 
                                             self.dsortedvalues,
                                             nbits,
                                             startbit,
                                             nelements,
                                             totalBlocks,
                                             self.local1).wait()
        
        # STEP II
        totalBlocks = numpy.uint32(nelements/2/ctaSize)
        globalWorkSize = (numpy.int(ctaSize*totalBlocks),)
        localWorkSize = (numpy.int(ctaSize),)

        # create Local Memory
        self.local2 = cl.LocalMemory(size=numpy.int(2 * ctaSize * self.size_uint))

	self.program.findRadixOffsets(self.queue, globalWorkSize, localWorkSize,
                                      self.dsortedkeys,
                                      self.mCounters,
                                      self.mBlockOffsets,
                                      startbit,
                                      nelements,
                                      totalBlocks,
                                      self.local2).wait()
        
        # STEP III
	scan.scanExclusiveLarge(self.mCountersSum, self.mCounters, 1, nelements/2/ctaSize*16)
        
        # STEP IV
        totalBlocks = numpy.uint32(nelements/2/ctaSize)
        globalWorkSize = (numpy.int(ctaSize*totalBlocks),)
        localWorkSize = (numpy.int(ctaSize),)

        # create Local Memory
        self.local3 = cl.LocalMemory(size=numpy.int(2 * ctaSize * self.size_uint))
        self.local4 = cl.LocalMemory(size=numpy.int(2 * ctaSize * self.size_uint))
        
	self.program.reorderDataKeysOnly(self.queue, globalWorkSize, localWorkSize,
                                         self.dkeys,
                                         self.dsortedkeys,
                                         self.dvalues,
                                         self.dsortedvalues,
                                         self.mBlockOffsets,
                                         self.mCountersSum,
                                         self.mCounters,
                                         startbit,
                                         nelements,
                                         totalBlocks,
                                         self.local3,
                                         self.local4).wait()

    

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

            # _keys and _values are the  padded arrays we use internally
            self._keys = numpy.concatenate( (self.keys,
                                             pad) ).astype(numpy.uint32)

            self._values = numpy.concatenate( (self.values,
                                               pad) ).astype(numpy.uint32)
        else:
            self._keys = self.keys
            self._values = self.values

        # now store the number of elements and num work groups
        self.nelements = numpy.uint32(len(self._keys))

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
        nelements = self.nelements
                                                   
        WARP_SIZE = 32

        self.size_uint = size_uint = numpy.uint32(0).nbytes
        self.keyBits = keybits = numpy.uint32(32)
        self.bitStep = numpy.uint32(4)

        if (nelements>=4096*4):
            ctaSize = 128
        elif(nelements<4096*4 and nelements>2048):
            ctaSize = 64
        else:
            raise RuntimeError( "particles < 4096")
            
        self.ctaSize = ctaSize
        
        if ((nelements % (ctaSize * 4)) == 0):
            numBlocks = numpy.uint32(nelements/(ctaSize * 4))
        else:
            numBlocks = numpy.uint32(nelements/(ctaSize * 4) + 1)

        # first allocate the keys and values on the device
        # these serve as the unsorted keys and values on the device
        self.dkeys = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR,
                               hostbuf=self._keys)

        self.dvalues = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR,
                                 hostbuf=self._values)

        # the final output or the sorted output.
        # This should obviously be of size num_elements
        self.sortedkeys = numpy.ones(self.nelements, dtype=numpy.uint32)
        self.sortedvalues = numpy.ones(self.nelements, dtype=numpy.uint32)

        # Buffers 
        self.dsortedkeys = cl.Buffer(ctx, mf.READ_WRITE, numpy.int(size_uint * self.nelements))
        self.dsortedvalues = cl.Buffer(ctx, mf.READ_WRITE, numpy.int(size_uint * self.nelements))
        
        self.mCounters = cl.Buffer(ctx, mf.READ_WRITE, numpy.int(WARP_SIZE * size_uint * numBlocks))
        self.mCountersSum = cl.Buffer(ctx, mf.READ_WRITE, numpy.int(WARP_SIZE * size_uint * numBlocks))
        self.mBlockOffsets = cl.Buffer(ctx, mf.READ_WRITE, numpy.int(WARP_SIZE * size_uint * numBlocks))


    def _create_program(self):
        """Read the OpenCL kernel file and build"""
        src_file = clu.get_pysph_root() + '/base/RadixSortVal.cl'
        src = open(src_file).read()
        self.program = cl.Program(self.context, src).build()


        
        
