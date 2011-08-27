""" An implementation of a general solver base class """

# PySPH imports
from pysph.base.particle_types import ParticleType
from pysph.base.carray import LongArray
from pysph.base.kernels import CubicSplineKernel
from pysph.base.particle_array import get_particle_array

from pysph.sph.kernel_correction import KernelCorrectionManager
from pysph.sph.sph_calc import SPHCalc, CLCalc
from pysph.sph.funcs.position_funcs import PositionStepping
from pysph.sph.funcs.xsph_funcs import XSPHCorrection

from sph_equation import SPHOperation, SPHIntegration
from integrator import EulerIntegrator
from cl_integrator import CLEulerIntegrator
from utils import PBar, savez_compressed, savez, load
from cl_utils import get_cl_devices, HAS_CL, create_some_context
from time_step_functions import TimeStep

if HAS_CL:
    import pyopencl as cl

import logging
logger = logging.getLogger()

import os
import sys
import numpy

Fluids = ParticleType.Fluid

class Solver(object):
    """ Base class for all PySPH Solvers

    **Attributes**
    
    - particles -- the particle arrays to operate on

    - integrator_type -- the class of the integrator. This may be one of any 
      defined in solver/integrator.py

    - kernel -- the kernel to be used throughout the calculations. This may 
      need to be modified to handle several kernels.

    - operation_dict -- an internal structure indexing the operation id and 
      the corresponding operation as a dictionary

    - order -- a list of strings specifying the order of an SPH simulation.
    
    - t -- the internal time step counter

    - pre_step_functions -- a list of functions to be performed before stepping

    - post_step_functions -- a list of functions to execute after stepping

    - pfreq -- the output print frequency

    - dim -- the dimension of the problem

    - kernel_correction -- flag to indicate type of kernel correction.
      Defaults to -1 for no correction

    - pid -- the processor id if running in parallel

    - eps -- the epsilon value to use for XSPH stepping. 
      Defaults to -1 for no XSPH

    """
    
    def __init__(self, dim, integrator_type):

        self.integrator_type = integrator_type
        self.dim = dim
        self.eps = -1

        self.cl_integrator_types = {EulerIntegrator:CLEulerIntegrator}

        self.initialize()

    def initialize(self):
        """ Perform basic initializations """

        # set the particles to None
        self.particles = None

        # default SPH kernel
        self.default_kernel = CubicSplineKernel(dim=self.dim)

        # flag to use OpenCL
        self.with_cl = False

        # mapping between operation id's and SPHOperations
        self.operation_dict = {}

        # Order of precedence for the SPHOperations
        self.order = []

        # solver time and iteration count
        self.t = 0
        self.count = 0

        self.execute_commands = None

        # list of functions to be called before and after an integration step
        self.pre_step_functions = []
        self.post_step_functions = []

        # default output printing frequency
        self.pfreq = 100

        # Integer identifying the type of kernel correction to use
        self.kernel_correction = -1

        # the process id for parallel runs
        self.pid = None

        # set the default rank to 0
        self.rank = 0

        # set the default mode to serial
        self.in_parallel = False

        # default function to dynamically compute time step
        self.time_step_function = TimeStep()

        # arrays to print output
        self.arrays_to_print = []

        # the default parallel output mode
        self.parallel_output_mode = "collected"

        # default particle properties to print
        self.print_properties = ['x','u','m','h','p','e','rho',]
        if self.dim > 1:
            self.print_properties.extend(['y','v'])

        if self.dim > 2:
            self.print_properties.extend(['z','w'])

        # flag to print all arrays 
        self.detailed_output = False

        # output filename
        self.fname = self.__class__.__name__

        # output drectory
        self.output_directory = self.fname+'_output'

    def switch_integrator(self, integrator_type):
        """ Change the integrator for the solver """

        if self.particles == None:
            raise RuntimeError, "There are no particles!"

        if self.with_cl:
            self.integrator_type = self.cl_integrator_types[integrator_type]

        else:
            self.integrator_type = integrator_type

        self.setup(self.particles)

    def add_operation_step(self, types, xsph=False, eps=0.5):
        
        """ Specify an acceptable list of types to step

        Parameters
        ----------
        types : a list of acceptable types eg Fluid, Solid

        Notes
        -----
        The types are defined in base/particle_types.py

        """
        updates = ['x','y','z'][:self.dim]

        id = 'step'
        
        self.add_operation(SPHIntegration(
            PositionStepping, on_types=types, updates=updates, id=id,
            kernel=None)
                           )

    def add_operation_xsph(self, eps, hks=False):
        """ Set the XSPH operation if requested

        Parameters
        ----------
        eps : the epsilon value to use for XSPH stepping
        
        Notes
        -----
        The position stepping operation must be defined. This is because
        the XSPH operation is setup for those arrays that need stepping.

        The smoothing kernel used for this operation is the CubicSpline

        """       
        
        assert eps > 0, 'Invalid value for XSPH epsilon: %f' %(eps)
        self.eps = eps

        # create the xsph stepping operation
                
        id = 'xsph'
        err = "position stepping function does not exist!"
        assert self.operation_dict.has_key('step'), err

        types = self.operation_dict['step'].on_types
        updates = self.operation_dict['step'].updates

                           
        self.add_operation(SPHIntegration(

            XSPHCorrection.withargs(eps=eps, hks=hks), from_types=types,
            on_types=types, updates=updates, id=id, kernel=self.default_kernel)

                           )

    def add_operation(self, operation, before=False, id=None):
        """ Add an SPH operation to the solver.

        Parameters
        ----------
        operation : the operation (:class:`SPHOperation`) to add
        before : flag to indicate insertion before an id. Defaults to False
        id : The id where to insert the operation. Defaults to None

        Notes
        -----
        An SPH operation typically represents a single equation written
        in SPH form. SPHOperation is defined in solver/sph_equation.py

        The id for the operation must be unique. An error is raised if an
        operation with the same id exists.

        Similarly, an error is raised if an invalid 'id' is provided 
        as an argument.

        Examples
        --------
        (1)        

        >>> solver.add_operation(operation)
        
        This appends an operation to the existing list. 

        (2)
        
        >>> solver.add_operation(operation, before=False, id=someid)
        
        Add an operation after an existing operation with id 'someid'

        (3)
        
        >>> solver.add_operation(operation, before=True, id=soleid)
        
        Add an operation before the operation with id 'someid'
        

        """
        err = 'Operation %s exists!'%(operation.id)
        assert operation.id not in self.order, err
        assert operation.id not in self.operation_dict.keys()

        self.operation_dict[operation.id] = operation
            
        if id:
            msg = 'The specified operation doesnt exist'
            assert self.operation_dict.has_key(id), msg  + ' in the calcs dict!'
            assert id in self.order, msg + ' in the order list!'

            if before:                
                self.order.insert(self.order.index(id), operation.id)

            else:
                self.order.insert(self.order.index(id)+1, operation.id)
            
        else:
            self.order.append(operation.id)

    def replace_operation(self, id, operation):
        """ Replace an operation.

        Parameters
        ----------
        id : the operation with id to replace
        operation : The replacement operation

        Notes
        -----
        The id to replace is taken from the provided operation. 
        
        An error is raised if the provided operation does not exist.

        """

        msg = 'The specified operation doesnt exist'
        assert self.operation_dict.has_key(id), msg  + ' in the op dict!'
        assert id in self.order, msg + ' in the order list!'

        self.operation_dict.pop(id)

        self.operation_dict[operation.id] = operation

        idx = self.order.index(id)
        self.order.insert(idx, operation.id)
        self.order.remove(id)

    def remove_operation(self, id_or_operation):
        """ Remove an operation with id

        Parameters
        ----------
        id_or_operation : the operation to remove

        Notes
        -----
        Remove an operation with either the operation object or the 
        operation id.

        An error is raised if the operation is invalid.
        
        """
        if type(id_or_operation) == str:
            id = id_or_operation
        else:
            id = id_or_operation.id

        assert id in self.operation_dict.keys(), 'id doesnt exist!'
        assert id in self.order, 'id doesnt exist!'

        self.order.remove(id)
        self.operation_dict.pop(id)

    def set_order(self, order):
        """ Install a new order 

        The order determines the manner in which the operations are
        executed by the integrator.

        The new order and existing order should match else, an error is raised

        """
        for equation_id in order:
            msg = '%s in order list does not exist!'%(equation_id)
            assert equation_id in self.order, msg
            assert equation_id in self.operation_dict.keys(), msg

        self.order = order

    def setup_position_step(self):
        """ Setup the position stepping for the solver """
        pass

    def setup(self, particles=None):
        """ Setup the solver.

        The solver's processor id is set if the in_parallel flag is set 
        to true.

        The order of the integrating calcs is determined by the solver's 
        order attribute.

        This is usually called at the start of a PySPH simulation.

        By default, the kernel correction manager is set for all the calcs.
        
        """
        
        if particles:
            self.particles = particles

            self.particles.kernel = self.default_kernel

            # instantiate the Integrator
            self.integrator = self.integrator_type(particles, calcs=[])

            # setup the SPHCalc objects for the integrator
            for equation_id in self.order:
                operation = self.operation_dict[equation_id]

                if operation.kernel is None:
                    operation.kernel = self.default_kernel

                calcs = operation.get_calcs(particles, operation.kernel)

                self.integrator.calcs.extend(calcs)

            if self.with_cl:
                self.integrator.setup_integrator(self.cl_context)
            else:
                self.integrator.setup_integrator()

            # Setup the kernel correction manager for each calc
            calcs = self.integrator.calcs
            particles.correction_manager = KernelCorrectionManager(
                calcs, self.kernel_correction)

    def add_print_properties(self, props):
        """ Add a list of properties to print """
        for prop in props:
            if not prop in self.print_properties:
                self.print_properties.append(prop)            

    def append_particle_arrrays(self, arrays):
        """ Append the particle arrays to the existing particle arrays """

        if not self.particles:
            print 'Warning!, particles not defined'
            return
        
        for array in self.particles.arrays:
            array_name = array.name
            for arr in arrays:
                if array_name == arr.name:
                    array.append_parray(arr)

        self.setup(self.particles)

    def set_final_time(self, tf):
        """ Set the final time for the simulation """
        self.tf = tf

    def set_time_step(self, dt):
        """ Set the time step to use """
        self.dt = dt

    def set_print_freq(self, n):
        """ Set the output print frequency """
        self.pfreq = n

    def set_arrays_to_print(self, array_names=None):

        available_arrays = [array.name for array in self.particles.arrays]
        
        if array_names:
            for name in array_names:

                if not name in available_arrays:
                    raise RuntimeError("Array %s not availabe"%(name))
                
                array = self.particles.get_named_particle_array(name)
                self.arrays_to_print.append(array)
        else:
            self.arrays_to_print = self.particles.arrays

    def set_output_fname(self, fname):
        """ Set the output file name """
        self.fname = fname

    def set_output_printing_level(self, detailed_output):
        """ Set the output printing level """
        self.detailed_output = detailed_output

    def set_output_directory(self, path):
        """ Set the output directory """
        self.output_directory = path

    def set_kernel_correction(self, kernel_correction):
        """ Set the kernel correction manager for each calc """
        self.kernel_correction = kernel_correction
        
        for id in self.operation_dict:
            self.operation_dict[id].kernel_correction=kernel_correction

    def set_parallel_output_mode(self, mode="collected"):
        """Set the default solver dump mode in parallel.

        The available modes are:

        collected : Collect array data from all processors on root and
                    dump a single file.


        distributed : Each processor dumps a file locally.

        """
        self.parallel_output_mode = mode

    def set_cl(self, with_cl=False):
        """ Set the flag to use OpenCL

        This option must be set after all operations are created so that
        we may switch the default SPHCalcs to CLCalcs.

        The solver must also setup an appropriate context which is used
        to setup the ParticleArrays on the device.
        
        """
        self.with_cl = with_cl

        if with_cl:
            if not HAS_CL:
                raise RuntimeWarning, "PyOpenCL not found!"

            for equation_id in self.order:
                operation = self.operation_dict[equation_id]

                # set the type of calc to use for the operation

                operation.calc_type = CLCalc

            # HACK. THE ONLY CL INTEGRATOR IS EULERINTEGRATOR

            #self.integrator_type = self.cl_integrator_types[
            #    self.integrator_type]
            self.integrator_type = CLEulerIntegrator

            # Setup the OpenCL context
            self.setup_cl()

    def set_command_handler(self, callable, command_interval=1):
        """ set the `callable` to be called at every `command_interval` iteration
        
        the `callable` is called with the solver instance as an argument
        """
        self.execute_commands = callable
        self.command_interval = command_interval

    def solve(self, show_progress=False):
        """ Solve the system

        Notes
        -----
        Pre-stepping functions are those that need to be called before
        the integrator is called. 

        Similarly, post step functions are those that are called after
        the stepping within the integrator.

        """
        dt = self.dt

        bt = (self.tf - self.t)/1000.0
        bcount = 0.0
        bar = PBar(1001, show=show_progress)

        self.dump_output(dt, *self.print_properties)

        # set the time for the integrator
        self.integrator.time = self.t

        while self.t < self.tf:
            self.t += dt
            self.count += 1

            # update the particles explicitly
            self.particles.update()

            # perform any pre step functions
            for func in self.pre_step_functions:
                func.eval(self)

            # compute the local time step
            if not self.with_cl:
                dt = self.time_step_function.compute_time_step(self)

            # compute the global time step
            dt = self.compute_global_time_step(dt)
            
            logger.info("Time %f, time step %f, rank  %d"%(self.t, dt,
                                                           self.rank))
            # perform the integration and update the time
            self.integrator.integrate(dt)
            self.integrator.time += dt

            # update the time for all arrays
            self.update_particle_time()

            # perform any post step functions            
            for func in self.post_step_functions:
                func.eval(self)

            # dump output
            if self.count % self.pfreq == 0:
                self.dump_output(dt, *self.print_properties)

            bcount += self.dt/bt
            while bcount > 0:
                bar.update()
                bcount -= 1
        
            if self.execute_commands is not None:
                if self.count % self.command_interval == 0:
                    self.execute_commands(self)

        bar.finish()

    def update_particle_time(self):
        for array in self.particles.arrays:
            array.set_time(self.t)

    def compute_global_time_step(self, dt):

        if self.particles.in_parallel:
            props = {'dt':dt}
            glb_min, glb_max = self.particles.get_global_min_max(props)
            return glb_min['dt']
        else:
            return dt

    def dump_output(self, dt, *print_properties):
        """ Print output based on level of detail required
        
        The default detail level (low) is the integrator's calc's update 
        property for each named particle array.
        
        The higher detail level dumps all particle array properties.

        Format:
        -------

        A single file named as: <fname>_<rank>_<count>.npz

        The output file contains the following fields:

        solver_data : Solver related data like time step, time and
        iteration count. These are used to resume a simulation.

        arrays : A dictionary keyed on particle array names and with
        particle properties as value.

        version : The version number for this format of file
        output. The current version number is 1

        Example:
        --------

        data = load('foo.npz')

        version = data['version']

        dt = data['solver_data']['dt']
        t = data['solver_data']['t']
        
        array = data['arrays'][array_name].astype(object)
        array['x']

        """

        if self.with_cl:
            self.particles.read_from_buffer()

        fname = self.fname + '_' 
        props = {"arrays":{}, "solver_data":{}}

        cell_size = None
        if not self.with_cl:
            cell_size = self.particles.cell_manager.cell_size

        _fname = os.path.join(self.output_directory,
                              fname  + str(self.count) +'.npz')

        if self.detailed_output:
            for array in self.particles.arrays:
                props["arrays"][array.name]=array.get_property_arrays(all=True)
        else:
            for array in self.particles.arrays:
                props["arrays"][array.name]=array.get_property_arrays(all=False)

        # Add the solver data
        props["solver_data"]["dt"] = dt
        props["solver_data"]["t"] = self.t
        props["solver_data"]["count"] = self.count

        if self.parallel_output_mode == "collected" and self.in_parallel:

            comm = self.comm
            
            arrays = props["arrays"]
            numarrays = len(arrays)
            array_names = arrays.keys()

            # gather the data from all processors
            collected_data = comm.gather(arrays, root=0)
            
            if self.rank == 0:
                props["arrays"] = {}

                size = comm.Get_size()

                # concatenate the arrays
                for array_name in array_names:
                    props["arrays"][array_name] = {}

                    _props = collected_data[0][array_name].keys()

                    for prop in _props:
                        prop_arr = numpy.concatenate( [collected_data[pid][array_name][prop] for pid in range(size)] )
                        
                        props["arrays"][array_name][prop] = prop_arr

                savez(_fname, version=1, **props)

        else:
            savez(_fname, version=1, **props)

    def load_output(self, count):
        """ Load particle data from dumped output file.

        Parameters
        ----------
        count : string
            The iteration time from which to load the data. If time is
            '?' then list of available data files is returned else 
             the latest available data file is used

        Notes
        -----
        Data is loaded from the :py:attr:`output_directory` using the same format
        as stored by the :py:meth:`dump_output` method.
        Proper functioning required that all the relevant properties of arrays be
        dumped

        """
        # get the list of available files
        available_files = [i.rsplit('_',1)[1][:-4] for i in os.listdir(self.output_directory) if i.startswith(self.fname) and i.endswith('.npz')]

        if count == '?':
            return sorted(set(available_files), key=int)

        else:
            if not count in available_files:
                msg = """File with iteration count `%s` does not exist"""%(count)
                msg += "\nValid iteration counts are %s"%(sorted(set(available_files), key=int))
                #print msg
                raise IOError(msg)

        array_names = [pa.name for pa in self.particles.arrays]

        # load the output file
        data = load(os.path.join(self.output_directory,
                                 self.fname+'_'+str(count)+'.npz'))
        
        arrays = [ data["arrays"][i] for i in array_names ]

        # set the Particle's arrays
        self.particles.arrays = arrays

        # call the particle's initialize
        self.particles.initialize()

        self.t = float(data["solver_data"]['t'])
        self.count = int(data["solver_data"]['count'])

    def setup_cl(self):
        """ Setup the OpenCL context and other initializations """
        if HAS_CL:
            self.cl_context = create_some_context()

    def get_options(self, opt_parser):
        """ Implement this to add additional options for the application """
        pass

    def setup_solver(self, options=None):
        """ Implement the basic solvers here 

        All subclasses of Solver may implement this function to add the 
        necessary operations for the problem at hand.

        Look at solver/fluid_solver.py for an example.

        Parameters
        ----------
        options : dict
            options set by the user using commandline (there is no guarantee
            of existence of any key)
        """
        pass 

############################################################################
