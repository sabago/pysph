# OpenCL conditional imports
import pysph.solver.cl_utils as clu

if clu.HAS_CL:
    import pyopencl as cl
    mf = cl.mem_flags

import numpy as np

class Scan(object):
    def __init__(self, GPUContext,
                 CommandQueue,
                 numElements):

        # Constants

        MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE = 1024
        MAX_LOCAL_GROUP_SIZE = 256
        self.WORKGROUP_SIZE = 256
	self.MAX_BATCH_ELEMENTS = 64 * 1048576; #64 * numElements
        self.MIN_SHORT_ARRAY_SIZE = 4;
        self.MAX_SHORT_ARRAY_SIZE = 4 * self.WORKGROUP_SIZE;
        self.MIN_LARGE_ARRAY_SIZE = 8 * self.WORKGROUP_SIZE;
        self.MAX_LARGE_ARRAY_SIZE = 4 * self.WORKGROUP_SIZE * self.WORKGROUP_SIZE;
        self.size_uint = size_uint = np.uint32(0).nbytes

        # OpenCL elements
        self.cxGPUContext = GPUContext
        self.cqCommandQueue = CommandQueue
        self.mNumElements = numElements
        mf = cl.mem_flags

        if (numElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE):
            self.d_Buffer = cl.Buffer(self.cxGPUContext, mf.READ_WRITE, np.int(numElements/MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE * size_uint))

        # Program
        src_file = clu.get_pysph_root() + '/base/Scan_b.cl'
        src = open(src_file).read()
        cpProgram = cl.Program(self.cxGPUContext, src).build()

        
        # Kernel
        self.ckScanExclusiveLocal1 = cpProgram.scanExclusiveLocal1
        self.ckScanExclusiveLocal2 = cpProgram.scanExclusiveLocal2
        self.ckUniformUpdate = cpProgram.uniformUpdate

    def scanExclusiveLarge(self, d_Dst, d_Src, batchSize, arrayLength):
        # I
        WORKGROUP_SIZE = self.WORKGROUP_SIZE
        size = np.uint32(4 * WORKGROUP_SIZE)
        n = (batchSize * arrayLength) / (4 * WORKGROUP_SIZE)
        localWorkSize = (np.int(WORKGROUP_SIZE),)
        globalWorkSize = (np.int((n * size) / 4), )

        # create Local Memory
        l_data1 = cl.LocalMemory(np.int(2 * WORKGROUP_SIZE * self.size_uint))

        self.ckScanExclusiveLocal1(self.cqCommandQueue, globalWorkSize, localWorkSize,
                                   d_Dst,
                                   d_Src,
                                   l_data1,
                                   size).wait()
  
        # II
        size = np.uint32(arrayLength / (4 * WORKGROUP_SIZE))
        n = batchSize
        elements = np.uint32(n * size)
        globalWorkSize = (self.iSnapUp(elements, WORKGROUP_SIZE),)
        
        # create Local Memory
        l_data2 = cl.LocalMemory(np.int(2 * WORKGROUP_SIZE * self.size_uint))

        self.ckScanExclusiveLocal2(self.cqCommandQueue, globalWorkSize, localWorkSize,
                                   self.d_Buffer,
                                   d_Dst,
                                   d_Src,
                                   l_data2,
                                   elements,
                                   size).wait()


        # III
        n = (batchSize * arrayLength) / (4 * WORKGROUP_SIZE)
        localWorkSize = (np.int(WORKGROUP_SIZE),)
        globalWorkSize = (np.int(n * WORKGROUP_SIZE),)
        
        self.ckUniformUpdate(self.cqCommandQueue, globalWorkSize, localWorkSize,
                             d_Dst,
                             self.d_Buffer).wait()


    def iSnapUp(self, dividend, divisor):
        rem = dividend%divisor
        if (rem == 0):
            return np.int(dividend)
        else:
            return np.int(dividend - rem + divisor)
