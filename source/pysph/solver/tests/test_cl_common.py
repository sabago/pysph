""" Tests for the OpenCL functions in cl_common """

import numpy
import pysph.solver.api as solver

if solver.HAS_CL:
    import pyopencl as cl

else:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass


from os import path

np = 16*16*16

x = numpy.ones(np, numpy.float32)
y = numpy.ones(np, numpy.float32)
z = numpy.ones(np, numpy.float32)

platform = cl.get_platforms()[0]
devices = platform.get_devices()
device = devices[0]

ctx = cl.Context(devices)
q = cl.CommandQueue(ctx, device)

mf = cl.mem_flags

xbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
ybuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
zbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=z)

args = (ybuf, zbuf)

pysph_root = solver.get_pysph_root()
src = solver.cl_read(path.join(pysph_root, 'solver/cl_common.cl'),
                     precision='single')

prog = cl.Program(ctx, src).build(options=solver.get_cl_include())

# launch the OpenCL kernel
prog.set_tmp_to_zero(q, (16, 16, 16), (1,1,1), xbuf, *args)

# read the buffer contents back to the arrays
solver.enqueue_copy(q, src=xbuf, dst=x)
solver.enqueue_copy(q, src=ybuf, dst=y)
solver.enqueue_copy(q, src=zbuf, dst=z)

for i in range(np):
    assert x[i] == 0.0
    assert y[i] == 0.0
    assert z[i] == 0.0

print "OK"
