import tempfile
import os

import pysph.solver.cl_utils as clu

if clu.HAS_CL:
    import pyopencl as cl

else:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass
    

def test_cl_read():
    """Test if the pysph.solcer.cl_utils.cl_read works."""

    # Create a test file.
    fd, name = tempfile.mkstemp(suffix='.cl')
    code = """
    REAL foo = 1.0;
    """
    f = open(name, 'w')
    f.write(code)
    f.close()
    os.close(fd)

    # Test single precision
    src = clu.cl_read(name, precision='single')
    expect = """#define F f
#define REAL float
#define REAL2 float2
#define REAL3 float3
#define REAL4 float4
#define REAL8 float8
"""
    s_lines = src.split()
    for idx, line in enumerate(expect.split()):
        assert line == s_lines[idx]


    # Test double precision
    src = clu.cl_read(name, precision='double')
    expect = """#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define F
#define REAL double
#define REAL2 double2
#define REAL3 double3
#define REAL4 double4
#define REAL8 double8
"""
    s_lines = src.split()
    for idx, line in enumerate(expect.split()):
        assert line == s_lines[idx]

    # cleanup.
    os.remove(name)

def test_round_up():
    """Test the rounding up function"""

    # got any better ideas?
    assert clu.round_up(1000) == 1024
    assert clu.round_up(500) == 512

if __name__ == '__main__':
    test_cl_read()

