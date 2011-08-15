# Standard imports.
import logging, os
from optparse import OptionParser, OptionGroup, Option
from os.path import basename, splitext, abspath
import sys

from utils import mkdir

# PySPH imports.
from pysph.base.particles import Particles, CLParticles, ParticleArray
from pysph.solver.controller import CommandManager
from pysph.solver.integrator import integration_methods
from pysph.base.nnps import NeighborLocatorType as LocatorType
import pysph.base.kernels as kernels

# MPI conditional imports
HAS_MPI = True
try:
    from mpi4py import MPI
except ImportError:
    HAS_MPI = False
else:
    from pysph.parallel.load_balancer import LoadBalancer
    from pysph.parallel.simple_parallel_manager import \
            SimpleParallelManager
    from pysph.parallel.parallel_cell import ParallelCellManager
    from pysph.parallel.simple_block_manager import SimpleBlockManager

def list_option_callback(option, opt, value, parser):
    val = value.split(',')
    val.extend( parser.rargs )
    setattr( parser.values, option.dest, val )

##############################################################################
# `Application` class.
############################################################################## 
class Application(object):
    """ Class used by any SPH application """

    def __init__(self, load_balance=True, fname=None):
        """ Constructor

        Parameters
        ----------
        load_balance : A boolean which determines if automatic load
                          balancing is to be performed or not

        """
        self._solver = None
        self._parallel_manager = None

        # The initial distribution method name to pass to the LoadBalancer's
        # `distribute_particles` method, can be one of ('auto', 'sfc', 'single'
        # etc.)
        self._distr_func = 'auto'
        self.load_balance = load_balance

        if fname == None:
            fname = sys.argv[0].split('.')[0]

        self.fname = fname

        self.args = sys.argv[1:]

        # MPI related vars.
        self.comm = None
        self.num_procs = 1
        self.rank = 0
        if HAS_MPI:
            self.comm = comm = MPI.COMM_WORLD
            self.num_procs = comm.Get_size()
            self.rank = comm.Get_rank()
        
        self._log_levels = {'debug': logging.DEBUG,
                           'info': logging.INFO,
                           'warning': logging.WARNING,
                           'error': logging.ERROR,
                           'critical': logging.CRITICAL,
                           'none': None}

        self._setup_optparse()

        self.path = None
    
    def _setup_optparse(self):
        usage = """
        %prog [options] 

        Note that you may run this program via MPI and the run will be
        automatically parallelized.  To do this run::

         $ mpirun -n 4 /path/to/your/python %prog [options]
   
        Replace '4' above with the number of processors you have.
        Below are the options you may pass.

        """
        parser = OptionParser(usage)
        self.opt_parse = parser

        # Add some default options.
        parser.add_option("-b", "--no-load-balance", action="store_true",
                          dest="no_load_balance", default=False,
                          help="Do not perform automatic load balancing "\
                          "for parallel runs.")
        # -v
        valid_vals = "Valid values: %s"%self._log_levels.keys()
        parser.add_option("-v", "--loglevel", action="store",
                          type="string",
                          dest="loglevel",
                          default='warning',
                          help="Log-level to use for log messages. " +
                               valid_vals)
        # --logfile
        parser.add_option("--logfile", action="store",
                          type="string",
                          dest="logfile",
                          default=None,
                          help="Log file to use for logging, set to "+
                               "empty ('') for no file logging.")
        # -l 
        parser.add_option("-l", "--print-log", action="store_true",
                          dest="print_log", default=False,
                          help="Print log messages to stderr.")
        # --final-time
        parser.add_option("--final-time", action="store",
                          type="float",
                          dest="final_time",
                          default=None,
                          help="Total time for the simulation.")
        # --timestep
        parser.add_option("--timestep", action="store",
                          type="float",
                          dest="time_step",
                          default=None,
                          help="Timestep to use for the simulation.")
        # -q/--quiet.
        parser.add_option("-q", "--quiet", action="store_true",
                         dest="quiet", default=False,
                         help="Do not print any progress information.")

        # -o/ --output
        parser.add_option("-o", "--output", action="store",
                          dest="output", default=self.fname,
                          help="File name to use for output")

        # --output-freq.
        parser.add_option("--freq", action="store",
                          dest="freq", default=20, type="int",
                          help="Printing frequency for the output")
        
        # -d/ --detailed-output.
        parser.add_option("-d", "--detailed-output", action="store_true",
                         dest="detailed_output", default=False,
                         help="Dump detailed output.")

        # --directory
        parser.add_option("--directory", action="store",
                         dest="output_dir", default=self.fname+'_output',
                         help="Dump output in the specified directory.")

        # --kernel
        parser.add_option("--kernel", action="store",
                          dest="kernel", type="int",
                          help="%-55s"%"The kernel function to use:"+
                          ''.join(['%d - %-51s'%(d,s) for d,s in
                                     enumerate(kernels.kernel_names)]))

        # --hks
        parser.add_option("--hks", action="store_true",
                          dest="hks", default=True,
                          help="""Perform the Hrenquist and Katz kernel
                          normalization for variable smothing lengths.""")

        # -k/--kernel-correction
        parser.add_option("-k", "--kernel-correction", action="store",
                          dest="kernel_correction", type="int",
                          default=-1,
                          help="""Use Kernel correction.
                                  0 - Bonnet and Lok correction
                                  1 - RKPM first order correction""")

        # --integration
        parser.add_option("--integration", action="store",
                          dest="integration", type="int",
                          help="%-55s"%"The integration method to use:"+
                          ''.join(['%d - %-51s'%(d,s[0]) for d,s in
                                     enumerate(integration_methods)]))

        # --cl
        parser.add_option("--cl", action="store_true", dest="with_cl",
                          default=False, help=""" Use OpenCL to run the
                          simulation on an appropriate device """)

        # --parallel-mode
        parser.add_option("--parallel-mode", action="store",
                          dest="parallel_mode", default="simple",
                          help = """Use 'simple' (which shares all particles) 
                          or 'auto' (which does block based parallel
                          distribution of particles).""")

        # --parallel-output-mode
        parser.add_option("--parallel-output-mode", action="store",
                          dest="parallel_output_mode", default="collected",
                          help="""Use 'collected' to dump one output at
                          root or 'distributed' for every processor. """)


        # solver interfaces
        interfaces = OptionGroup(parser, "Interfaces",
                                 "Add interfaces to the solver")

        interfaces.add_option("--interactive", action="store_true",
                              dest="cmd_line", default=False,
                              help=("Add an interactive commandline interface "
                                    "to the solver"))
        
        interfaces.add_option("--xml-rpc", action="store",
                              dest="xml_rpc", metavar='[HOST:]PORT',
                              help=("Add an XML-RPC interface to the solver; "
                                    "HOST=0.0.0.0 by default"))
        
        interfaces.add_option("--multiproc", action="store",
                              dest="multiproc", metavar='[[AUTHKEY@]HOST:]PORT[+]',
                              default="pysph@0.0.0.0:8800+",
                              help=("Add a python multiprocessing interface "
                                    "to the solver; "
                                    "AUTHKEY=pysph, HOST=0.0.0.0, PORT=8800+ by"
                                    " default (8800+ means first available port "
                                    "number 8800 onwards)"))
        
        interfaces.add_option("--no-multiproc", action="store_const",
                              dest="multiproc", const=None,
                              help=("Disable multiprocessing interface "
                                    "to the solver"))
        
        parser.add_option_group(interfaces)
        
        # solver job resume support
        parser.add_option('--resume', action='store', dest='resume',
                          metavar='COUNT|count|?',
                          help=('Resume solver from specified time (as stored '
                                'in the data in output directory); count chooses '
                                'a particular file; ? lists all '
                                'available files')
                          )

    def _process_command_line(self):
        """ Parse any command line arguments.

        Add any new options before this is called.  This also sets up
        the logging automatically.

        """
        (options, args) = self.opt_parse.parse_args(self.args)
        self.options = options
        
        # Setup logging based on command line options.
        level = self._log_levels[options.loglevel]

        #save the path where we want to dump output
        self.path = abspath(options.output_dir)
        mkdir(self.path)

        if level is not None:
            self._setup_logging(options.logfile, level,
                                options.print_log)

    def _setup_logging(self, filename=None, loglevel=logging.WARNING,
                       stream=True):
        """ Setup logging for the application.
        
        Parameters
        ----------
        filename : The filename to log messages to.  If this is None
                   a filename is automatically chosen and if it is an
                   empty string, no file is used

        loglevel : The logging level

        stream : Boolean indicating if logging is also printed on
                    stderr
        """
        # logging setup
        self.logger = logger = logging.getLogger()
        logger.setLevel(loglevel)

        # Setup the log file.
        if filename is None:
            filename = splitext(basename(sys.argv[0]))[0] + '.log'

        if len(filename) > 0:
            lfn = os.path.join(self.path,filename)
            if self.num_procs > 1:
                logging.basicConfig(level=loglevel, filename=lfn,
                                    filemode='w')
        if stream:
            logger.addHandler(logging.StreamHandler())

    def _create_particles(self, variable_h, callable, min_cell_size=-1,
                         *args, **kw):
        """ Create particles given a callable and any arguments to it.
        This will also automatically distribute the particles among
        processors if this is a parallel run.  Returns the `Particles`
        instance that is created.
        """

        num_procs = self.num_procs
        rank = self.rank
        data = None
        if rank == 0:
            # Only master creates the particles.
            pa = callable(*args, **kw)
            distr_func = self._distr_func
            if num_procs > 1:
                # Use the offline load-balancer to distribute the data
                # initially. Negative cell size forces automatic computation. 
                data = LoadBalancer.distribute_particles(pa, 
                                                         num_procs=num_procs, 
                                                         block_size=-1, 
                                                         distr_func=distr_func)
        if num_procs > 1:
            # Now scatter the distributed data.
            pa = self.comm.scatter(data, root=0)

        self.particle_array = pa

        in_parallel = num_procs > 1
        if isinstance(pa, (ParticleArray,)):
            pa = [pa]

        no_load_balance = self.options.no_load_balance
        if no_load_balance:
            self.load_balance = False
        else:
            self.load_balance = True

        if self.options.with_cl:

            cl_locator_type = kw.get('cl_locator_type', None)
            domain_manager_type = kw.get('domain_manager_type', None)

            if cl_locator_type and domain_manager_type:

                self.particles = CLParticles(
                    arrays=pa, cl_locator_type=cl_locator_type,
                    domain_manager_type=domain_manager_type)

            else:
                self.particles = CLParticles(arrays=pa)
                
        else:

            locator_type = kw.get('locator_type', None)

            if locator_type:
                if locator_type not in [LocatorType.NSquareNeighborLocator,
                                        LocatorType.SPHNeighborLocator]:

                    msg = "locator type %d not understood"%(locator_type)
                    raise RuntimeError(msg)

            else:
                locator_type = LocatorType.SPHNeighborLocator

            self.particles = Particles(arrays=pa, variable_h=variable_h,
                                       in_parallel=in_parallel,
                                       load_balancing=self.load_balance,
                                       update_particles=True,
                                       min_cell_size=min_cell_size,
                                       locator_type=locator_type)

        return self.particles

    ######################################################################
    # Public interface.
    ###################################################################### 
    def set_args(self, args):
        self.args = args

    def add_option(self, opt):
        """ Add an Option/OptionGroup or their list to OptionParser """
        if isinstance(opt, OptionGroup):
            self.opt_parse.add_option_group(opt)
        elif isinstance(opt, Option):
            self.opt_parse.add_option(opt)
        else:
            # assume a list of Option/OptionGroup
            for o in opt:
                self.add_option(o)

    def setup(self, solver, create_particles=None,
              variable_h=False, min_cell_size=-1, **kwargs):
        """Set the application's solver.  This will call the solver's
        `setup` method.

        The following solver options are set:

        dt -- the time step for the solver

        tf -- the final time for the simulationl

        fname -- the file name for output file printing

        freq -- the output print frequency

        level -- the output detail level

        dir -- the output directory

        hks -- Hernquist and Katz kernel correction

        eps -- the xsph correction factor

        with_cl -- OpenCL related initializations

        integration_type -- The integration method

        default_kernel -- the default kernel to use for operations

        Parameters
        ----------
        create_particles : callable or None
            If supplied, particles will be created for the solver using the
            particle arrays returned by the callable. Else particles for the
            solver need to be set before calling this method

        variable_h : bool
            If the particles created using create_particles have variable h

        min_cell_size : float
            minimum cell size for particles created using min_cell_size
        """
        self._solver = solver
        solver_opts = solver.get_options(self.opt_parse)
        if solver_opts is not None:
            self.add_option(solver_opts)
        self._process_command_line()

        options = self.options

        if self.num_procs > 1:
            if options.parallel_mode == 'simple':
                self.set_parallel_manager(SimpleParallelManager())

            if options.parallel_mode == "block":
                self.set_parallel_manager( SimpleBlockManager() )

        if create_particles:
            self._create_particles(variable_h, create_particles, min_cell_size,
                                   **kwargs)

        pm = self._parallel_manager
        if pm is not None:
            self.particles.parallel_manager = pm
            pm.initialize(self.particles)

        self._solver.setup_solver(options.__dict__)

        dt = options.time_step
        if dt is not None:
            solver.set_time_step(dt)

        tf = options.final_time
        if tf is not None:
            solver.set_final_time(tf)

        #setup the solver output file name
        fname = options.output

        if HAS_MPI:
            comm = self.comm 
            rank = self.rank
            
            if not self.num_procs == 0:
                fname += '_' + str(rank)

        # set the rank for the solver
        solver.rank = self.rank
        solver.pid = self.rank
        solver.comm = self.comm

        # set the in parallel flag for the solver
        if self.num_procs > 1:
            solver.in_parallel = True

        # output file name
        solver.set_output_fname(fname)

        # output print frequency
        solver.set_print_freq(options.freq)

        # output printing level (default is not detailed)
        solver.set_output_printing_level(options.detailed_output)

        # output directory
        solver.set_output_directory(abspath(options.output_dir))

        # set parallel output mode
        solver.set_parallel_output_mode(options.parallel_output_mode)

        # default kernel
        if options.kernel is not None:
            solver.default_kernel = getattr(kernels,
                      kernels.kernel_names[options.kernel])(dim=solver.dim)

        # Hernquist and Katz kernel correction
        # TODO. Fix the Kernel and Gradient Correction
        #solver.set_kernel_correction(options.kernel_correction)

        # OpenCL setup for the solver
        solver.set_cl(options.with_cl)
        
        if options.resume is not None:
            solver.particles = self.particles # needed to be able to load particles
            r = solver.load_output(options.resume)
            if r is not None:
                print 'available files for resume:'
                print r
                sys.exit(0)

        if options.integration is not None:
            solver.integrator_type =integration_methods[options.integration][1]

        # setup the solver
        solver.setup(self.particles)

        # print options for the solver
        #solver.set_arrays_to_print(options.arrays_to_print)
        
        # add solver interfaces
        self.command_manager = CommandManager(solver, self.comm)
        solver.set_command_handler(self.command_manager.execute_commands)

        if self.rank == 0:
            # commandline interface
            if options.cmd_line:
                from pysph.solver.solver_interfaces import CommandlineInterface
                self.command_manager.add_interface(CommandlineInterface().start)
        
            # XML-RPC interface
            if options.xml_rpc:
                from pysph.solver.solver_interfaces import XMLRPCInterface
                addr = options.xml_rpc
                idx = addr.find(':')
                host = "0.0.0.0" if idx == -1 else addr[:idx]
                port = int(addr[idx+1:])
                self.command_manager.add_interface(XMLRPCInterface((host,port)).start)
        
            # python MultiProcessing interface
            if options.multiproc:
                from pysph.solver.solver_interfaces import MultiprocessingInterface
                addr = options.multiproc
                idx = addr.find('@')
                authkey = "pysph" if idx == -1 else addr[:idx]
                addr = addr[idx+1:]
                idx = addr.find(':')
                host = "0.0.0.0" if idx == -1 else addr[:idx]
                port = addr[idx+1:]
                if port[-1] == '+':
                    try_next_port = True
                    port = port[:-1]
                else:
                    try_next_port = False
                port = int(port)

                interface = MultiprocessingInterface((host,port), authkey,
                                                     try_next_port)

                self.command_manager.add_interface(interface.start)

                self.logger.info('started multiprocessing interface on %s'%(
                        interface.address,))

    def run(self):
        """Run the application."""
        self._solver.solve(not self.options.quiet)

    def set_parallel_manager(self, mgr):
        """Set the parallel manager class to use."""
        self._parallel_manager = mgr
        if isinstance(mgr, SimpleParallelManager):
            self._distr_func = 'auto'

