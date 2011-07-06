HAS_CL = True
try:
    import pyopencl as cl
except ImportError:
    HAS_CL=False

from os import path
import numpy

from utils import get_pysph_root

# Return all available devices on the host
def get_cl_devices():
    """ Return a dictionary keyed on device type for all devices """

    _devices = {'CPU':[], 'GPU':[]}

    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices()
        for device in devices:
            if device.type == cl.device_type.CPU:
                _devices['CPU'].append(device)
            elif device.type == cl.device_type.GPU:
                _devices['GPU'].append(device)
                
                
    return _devices

def create_context_from_cpu():
    """ Create an OpenCL context using the CPU as the default device """
    cl_devices = get_cl_devices()
    if ( cl_devices['CPU'] == [] ):
        raise ValueError("No CPU device found! ")

    return cl.Context( devices=cl_devices['CPU'] )

def create_context_from_gpu():
    """ Create an OpenCL context using the CPU as the default device """
    cl_devices = get_cl_devices()
    if ( cl_devices['GPU'] == [] ):
        raise ValueError("No CPU device found! ")

    return cl.Context( devices=cl_devices['GPU'] )

def create_some_context():
    """ Create a "reasonable" context from the available devices.

    Preference is given to CPU devices over GPU devices.

    """

    devices = get_cl_devices()

    cpu_devices = devices['CPU']
    gpu_devices = devices['GPU']

    if ( len( cpu_devices ) > 0 ):
        context = cl.Context( devices = cpu_devices )
        
    elif ( len( gpu_devices ) > 0 ):
        context = cl.Context( devices = gpu_devices )

    else:
        raise ValueError("No devices found!")

    return context

def get_cl_include():
    """ Include directories for OpenCL definitions """

    PYSPH_ROOT = get_pysph_root()

    if cl.version.VERSION_TEXT == "2011.1beta3":

        inc_dir = '-I'+path.join(PYSPH_ROOT, 'base') + " " + \
                  '-I'+path.join(PYSPH_ROOT, 'solver')

    elif cl.version.VERSION_TEXT == "2011.1.1":

        inc_dir = ["-I" + path.join(PYSPH_ROOT, "base"),
                   "-I" + path.join(PYSPH_ROOT, "solver") ]
        
    else:
        raise RuntimeWarning("Not supported yet")

    return inc_dir

def get_scalar_buffer(val, dtype, ctx):
    """ Return a cl.Buffer object that can be passed as a scalar to kernels """

    mf = cl.mem_flags

    arr = numpy.array([val,], dtype)
    return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)

def cl_read(filename, precision='double', function_name=None):
    """Read an OpenCL source file.  
    
    The function also adds a few convenient #define's so as to allow us
    to write common code for both float and double precision.  This is
    done by specifying the `precision` argument which defaults to
    'float'.  The OpenCL code itself should be written to use REAL for
    the type declaration.  The word REAL will be #defined to change
    precision on the fly.  For conveinence REAL2, REAL3, REAL4 and REAL8
    are all defined as well.

    Parameters
    ----------

    filename : str
        Name of file to open.

    precision : {'single', 'double'}, optional
        The floating point precision to use.

    function_name: str, optional
        An optional function name to indicate a block to extract from
        the OpenCL template file.

    """
    if precision not in ['single', 'double']:
        msg = "Invalid argument for 'precision' should be 'single'"\
              " or 'double'."
        raise ValueError(msg) 

    src = open(filename).read()

    if function_name:
        src = src.split('$'+function_name)[1]

    if precision == 'single':
        typ = 'float'
        hdr = "#define F f \n"
    else:
        typ = 'double'
        hdr = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        hdr += '#define F \n'

    for x in ('', '2', '3', '4', '8'):
        hdr += '#define REAL%s %%(typ)s%s\n'%(x, x)
    
    hdr = hdr%(dict(typ=typ))

    return hdr + src

def get_real(val, precision):
    """ Return a suitable floating point number for OpenCL.

    Parameters
    ----------

    val : float
        The value to convert.
        
    precision : {'single', 'double'}
        The precision to use.
    
    """
    if precision == "single":
        return numpy.float32(val)
    elif precision == "double":
        return numpy.float64(val)
    else:
        raise ValueError ("precision %s not supported!"%(precision))
    

def create_program(template, func, loc=None):
    """ Create an OpenCL program given a template string and function

    Parameters
    ----------

    template: str
        The template source file that is read using cl_read

    func: SPHFunctionParticle
        The function that provides the kernel arguments to the template

    loc: NotImplemented

    A template is the basic outline of an OpenCL kernel for a
    SPHFunctionParticle. The arguments to the kernel and neighbor
    looping code needs to be provided to render it a valid OpenCL
    kenrel.

    """

    k_args = []

    func.set_cl_kernel_args()
    k_args.extend(func.cl_args_name)

    # Build the kernel args string.
    kernel_args = ',\n    '.join(k_args)
    
    # Get the kernel workgroup code
    workgroup_code = func.get_cl_workgroup_code()
    
    # Construct the neighbor loop code.
    neighbor_loop_code = "for (int src_id=0; src_id<nbrs; ++src_id)"

    return template%(locals())

def enqueue_copy(queue, src, dst):

    if cl.version.VERSION_TEXT == "2011.1beta3":

        if ( isinstance(dst, cl.Buffer) ):

            if ( isinstance(src, cl.Buffer) ):

                # device to device copy
                cl.enqueue_copy_buffer(queue, src=src, dst=dst)

            elif ( isinstance(src, numpy.ndarray) ):

                # host to device copy
                cl.enqueue_write_buffer(queue, mem=dst, hostbuf=src)

        elif ( isinstance(src, cl.Buffer) ):

            cl.enqueue_read_buffer(queue, mem=src, hostbuf=dst)

    elif cl.version.VERSION_TEXT == "2011.1.1":
        cl.enqueue_copy(queue, dest=dst, src=src).wait()

    queue.finish()
