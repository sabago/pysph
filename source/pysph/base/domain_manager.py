import numpy

from pysph.solver.cl_utils import cl_read, get_real, HAS_CL, get_pysph_root,\
    create_some_context, enqueue_copy

if HAS_CL:
    import pyopencl as cl
    mf = cl.mem_flags

# Cython functions for neighbor list construction
from linked_list_functions import cbin

from point import Point
from cell import py_find_cell_id

class DomainManagerType:
    DomainManager = 0
    LinkedListManager = 1

class DomainManager:
    def __init__(self, arrays, context=None):
        if len(arrays) == 0:
            raise RuntimeError("No Arrays provided!")

        self.arrays = arrays
        self.narrays = narrays = len(arrays)

        # check if the arrays have unique names
        if narrays > 1:
            for i in range(1, narrays):
                if arrays[i].name == arrays[i-1].name:
                    msg = "You mnust provide arrays with unique names!"
                    raise RuntimeError(msg)

                if arrays[i].cl_precision != arrays[i-1].cl_precision:
                    msg = "Arrays cannot have different precision!"
                    raise RuntimeError(msg)

        # set the cl_precision
        self.cl_precision = arrays[0].cl_precision

        if HAS_CL:
            self.with_cl = True
            self.setup_cl(context)
        else:
            self.with_cl = False

    def setup_cl(self, context=None):
        """ OpenCL setup for the CLNNPSManager  """

        if not context:
            self.context = context = create_some_context()
            self.queue = queue = cl.CommandQueue(context)            
        else:
            self.context = context
            self.queue = queue = cl.CommandQueue(context)

        # allocate the particle array device buffers
        for i in range(self.narrays):
            pa = self.arrays[i]
            pa.setup_cl(context, queue)

    def update_status(self):
        """Updates the is_dirty flag to indicate that an update is required.

        Notes:
        ------

        Any module that may modify the particle data, should call this
        update_status function.

        """
        for i in range(self.narrays):
            parray = self.arrays[i]
            if parray.is_dirty:
                self.is_dirty = True
                break            

    def update(self):
        for pa in self.arrays:
            pa.set_dirty(False)

class LinkedListManager(DomainManager):
    """ Domain manager using bins as the indexing scheme and a linked
    list as the neighbor locator scheme.

    Data Attributes:
    ----------------

    arrays : list
         The particle arrays handled by the manager

    head : dict
           Head arrays for each ParticleArray maintained.
           The dictionary is keyed on name of the ParticleArray,
           with the head array as value.

    next : dict
           Next array for each ParticleArray maintained.
           The dictionary is keyed on name of the ParticleArray,
           with the next array as value.

    const_cell_size : REAL
        Optional constant cell size used for binning.
           
    cell_size :  REAL
           Cell size used for binning.

    cl_precision : string
           OpenCL precision to use. This is taken from the ParticleArrays

    Mx, mx, My, my, Mz, mz -- REAL
           Global bounds for the binning

    ncx, ncy, ncz -- uint
           Number of cells in each coordinate direction

    ncells -- uint
           Total number of cells : (ncx * ncy * ncz)

    is_dirty -- bool
           Flag to indicate an update us required.

    with_cl -- bool
           Flag to use OpenCL for the neighbor list generation.

    """

    def __init__(self, arrays, cell_size=None, context=None,
                 kernel_scale_factor = 2.0, with_cl=True):
        """ Construct a linked list manager.

        Parameters:
        ------------

        arrays -- list
                The ParticleArrays being managed.

        cell_size -- REAL
                The optional bin size to use

        kernel_scale_factor --REAL.
                the scale factor for the radius

        with_cl -- bool
            Explicitly choose OpenCL


        A LinkedListManager constructs and maintains a linked list for
        a list of particle arrays. The linked list data structure is
        consists of two arrays per particle array

        head : An integer array of size ncells, where ncells is the
        total number of cells in the domain. Each entry points to the
        index of a particle belonging to the cell. A negative index
        (-1) indicates and empty cell.

        next : An integer array of size num_particles. Each entry
        points to the next particle in the same cell. A negative index
        (-1) indicates no more particles. 

        The bin size, if provided is constant in each coordinate
        direction. The default choice for the bin size is twice the
        maximum smoothing length for all particles in the domain.

        """

        if len(arrays) == 0:
            raise RuntimeError("No Arrays provided!")

        self.arrays = arrays
        self.narrays = narrays = len(arrays)

        self.kernel_scale_factor = kernel_scale_factor

        # check if the arrays have unique names
        if narrays > 1:
            for i in range(1, narrays):
                if arrays[i].name == arrays[i-1].name:
                    msg = "You mnust provide arrays with unique names!"
                    raise RuntimeError(msg)

                if arrays[i].cl_precision != arrays[i-1].cl_precision:
                    msg = "Arrays cannot have different precision!"
                    raise RuntimeError(msg)

        # set the cl_precision
        self.cl_precision = arrays[0].cl_precision

        # set the cell size
        self.const_cell_size = cell_size
        if cell_size:
            self.const_cell_size = get_real(const_cell_size, self.cl_precision)

        # find global bounds (simulation box and ncells)
        self.find_bounds()

        # The linked list structures for the arrays.
        self.next = {}
        self.head = {}
        self.cellids = {}
        self.locks = {}

        self.ix = {}
        self.iy = {}
        self.iz = {}

        # initialize the dirty bit to true
        self.is_dirty = True

        # device linked list structures
        self.dnext = {}
        self.dhead = {}
        self.dcellids = {}
        self.dlocks = {}

        self.dix = {}
        self.diy = {}
        self.diz = {}

        if with_cl:
            if HAS_CL:
                self.with_cl = True
                self.setup_cl(context)
        else:
            self.with_cl = False

        # dict for kernel launch parameters 
        self.global_sizes = {}
        self.local_sizes = {}

        # initialize the linked list
        self.init_linked_list()

    def find_bounds(self):
        """ Find the bounds for the particle arrays.

        The bounds calculated are the simulation cube, defined by the
        minimum and maximum extents of the particle arrays and the
        maximum smoothing length which is used for determining an safe
        cell size for binning.

        """

        mx, my, mz = 1000.0, 1000.0, 1000.0
        Mx, My, Mz = -1000.0, -1000.0, -1000.0
        Mh = 0.0

        # update the minimum and maximum for the particle arrays
        for pa in self.arrays:
            pa.read_from_buffer()
            pa.update_min_max(props=['x','y','z','h'])

        if self.narrays > 0:

            if pa.properties['x'].minimum < mx:
                mx = get_real( pa.properties['x'].minimum, self.cl_precision )

            if pa.properties['y'].minimum < my:
                my = get_real( pa.properties['y'].minimum, self.cl_precision )

            if pa.properties['z'].minimum < mz:
                mz = get_real( pa.properties['z'].minimum, self.cl_precision )

            if pa.properties['x'].maximum > Mx:
                Mx = get_real( pa.properties['x'].maximum, self.cl_precision )

            if pa.properties['y'].maximum > My:
                My = get_real( pa.properties['y'].maximum, self.cl_precision )

            if pa.properties['z'].maximum > Mz:
                Mz = get_real( pa.properties['z'].maximum, self.cl_precision )

            if pa.properties['h'].maximum > Mh:
                Mh = get_real( pa.properties['h'].maximum, self.cl_precision )

        self.mx, self.my, self.mz = mx, my, mz
        self.Mx, self.My, self.Mz = Mx, My, Mz
        self.Mh = Mh

        self.set_cell_size()
        self.find_num_cells()

    def set_cell_size(self):
        """ Set the cell size for binning

        Parameters:
        -----------

        size -- An optional size to coerce the bin size

        Notes:
        ------

        If no bin sie is provided, the default value 2*max(h) is used

        """
        if not self.const_cell_size:
            self.cell_size = get_real(self.kernel_scale_factor*self.Mh,
                                      self.cl_precision)
        else:
            self.cell_size = self.const_cell_size

    def find_num_cells(self):
        """ Find the number of Cells in each coordinate direction

        The number of cells is found from the simulation bounds and
        the cell size used for binning.

        """

        max_pnt = Point(self.Mx, self.My, self.Mz)
        max_cid = py_find_cell_id(max_pnt, self.cell_size)

        min_pnt = Point(self.mx, self.my, self.mz)
        min_cid = py_find_cell_id(min_pnt, self.cell_size)

        self.ncx = numpy.int32(max_cid.x - min_cid.x + 1)
        self.ncy = numpy.int32(max_cid.y - min_cid.y + 1)
        self.ncz = numpy.int32(max_cid.z - min_cid.z + 1)

        self.mcx = numpy.int32(min_cid.x)
        self.mcy = numpy.int32(min_cid.y)
        self.mcz = numpy.int32(min_cid.z)

        self.ncells = numpy.int32(self.ncx * self.ncy * self.ncz)

    def init_linked_list(self):
        """ Initialize the linked list dictionaries to store the
        particle neighbor information.

        Three arrays, namely, head, next and cellids are created per
        particle array. 

        """
        ncells = self.ncells
        for i in range(self.narrays):
            pa = self.arrays[i]
            
            np = pa.get_number_of_particles()

            head = numpy.ones(ncells, numpy.int32) * numpy.int32(-1)
            next = numpy.ones(np, numpy.int32) * numpy.int32(-1)
            cellids = numpy.ones(np, numpy.uint32)
            locks = numpy.zeros(ncells, numpy.int32)

            ix = numpy.ones(np, numpy.uint32)
            iy = numpy.ones(np, numpy.uint32)
            iz = numpy.ones(np, numpy.uint32)

            self.head[pa.name] = head
            self.next[pa.name] = next
            self.cellids[pa.name] = cellids
            self.locks[pa.name] = locks

            self.ix[pa.name] = ix
            self.iy[pa.name] = iy
            self.iz[pa.name] = iz

        if self.with_cl:
            self.init_device_buffers()

    def init_device_buffers(self):
        """ Initialize the device buffers """

        for i in range(self.narrays):
            pa = self.arrays[i]
            np = pa.get_number_of_particles()

            # initialize the kerel launch parameters
            self.global_sizes[pa.name] = (np,)
            self.local_sizes[pa.name] = (1,)

            head = self.head[pa.name]
            next = self.next[pa.name]
            cellids = self.cellids[pa.name]
            locks = self.locks[pa.name]

            ix = self.ix[pa.name]
            iy = self.iy[pa.name]
            iz = self.iz[pa.name]

            dhead = cl.Buffer(self.context,
                              mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=head)

            dnext = cl.Buffer(self.context,
                              mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=next)

            dcellids = cl.Buffer(self.context,
                                 mf.READ_WRITE | mf.COPY_HOST_PTR,
                                 hostbuf=cellids)

            dlocks = cl.Buffer(self.context,
                               mf.READ_WRITE | mf.COPY_HOST_PTR,
                               hostbuf=locks)

            dix = cl.Buffer(self.context,
                            mf.READ_WRITE | mf.COPY_HOST_PTR,
                            hostbuf=ix)

            diy = cl.Buffer(self.context,
                            mf.READ_WRITE | mf.COPY_HOST_PTR,
                            hostbuf=iy)

            diz = cl.Buffer(self.context,
                            mf.READ_WRITE | mf.COPY_HOST_PTR,
                            hostbuf=iz)

            self.dhead[pa.name] = dhead
            self.dnext[pa.name] = dnext
            self.dcellids[pa.name] = dcellids
            self.dlocks[pa.name] = dlocks

            self.dix[pa.name] = dix
            self.diy[pa.name] = diy
            self.diz[pa.name] = diz

    def reset_data(self):
        """ Initialize the data structures.

        Head is initialized to -1
        Next is initialized to -1
        locks is initialized to 0

        """

        if self.with_cl:
            self.reset_cl_data()
        else:
            self.reset_cy_data()

    def reset_cy_data(self):
        for pa in self.arrays:
            head = self.head[pa.name]
            next = self.next[pa.name]

            head[:] = -1
            next[:] = -1

    def reset_cl_data(self):
        for pa in self.arrays:

            dhead = self.dhead[pa.name]
            dnext = self.dnext[pa.name]
            dlocks = self.dlocks[pa.name]

            global_sizes = (int(self.ncells),)
            val = numpy.int32(-1)

            self.prog.reset(self.queue, global_sizes, None, dhead, val).wait()

            val = numpy.int32(0)
            self.prog.reset(self.queue, global_sizes, None, dlocks, val).wait()

            global_sizes = self.global_sizes[pa.name]
            val = numpy.int32(-1)
            self.prog.reset(self.queue, global_sizes, None, dnext, val).wait()

    def cy_update(self):
        """ Construct the linked lists for the particle arrays using Cython"""

        ncx, ncy, ncz = self.ncx, self.ncy, self.ncz

        mx, my, mz = self.mx, self.my, self.mz
        cell_size = self.cell_size

        cell_size = get_real(self.cell_size, self.cl_precision)

        for i in range(self.narrays):
            pa = self.arrays[i]
            np = pa.get_number_of_particles()

            x, y, z = pa.get('x','y','z')
            if self.cl_precision == 'single':
                x = x.astype(numpy.float32)
                y = y.astype(numpy.float32)
                z = z.astype(numpy.float32)

            cbin( x, y, z,
                  self.cellids[pa.name],
                  self.ix[pa.name],
                  self.iy[pa.name],
                  self.iz[pa.name],
                  self.head[pa.name],
                  self.next[pa.name],
                  mx, my, mz,
                  numpy.int32(ncx), numpy.int32(ncy), numpy.int32(ncz),
                  cell_size, numpy.int32(np),
                  self.mcx, self.mcy, self.mcz
                  )


    def cl_update(self):
        """ Construct the linked lists for the particle arrays using OpenCL"""

        for i in range(self.narrays):
            pa = self.arrays[i]

            x = pa.get_cl_buffer('x')
            y = pa.get_cl_buffer('y')
            z = pa.get_cl_buffer('z')

            # Bin particles
            self.prog.bin( self.queue,
                           self.global_sizes[pa.name],
                           self.local_sizes[pa.name],
                           x, y, z,
                           self.dcellids[pa.name],
                           self.dix[pa.name],
                           self.diy[pa.name],
                           self.diz[pa.name],
                           self.mx,
                           self.my,
                           self.mz,
                           self.ncx,
                           self.ncy,
                           self.ncz,
                           self.cell_size,
                           self.mcx,
                           self.mcy,
                           self.mcz
                           ).wait()

            self.prog.construct_neighbor_list(self.queue,
                                              self.global_sizes[pa.name],
                                              self.local_sizes[pa.name],
                                              self.dcellids[pa.name],
                                              self.dhead[pa.name],
                                              self.dnext[pa.name],
                                              self.dlocks[pa.name]
                                              ).wait()
    def update(self):
        """ Update the linked list """

        if self.is_dirty:

            # find the bounds for the manager
            self.find_bounds()

            # reset the data structures
            #self.reset_data()
            self.init_linked_list()

            # update the data structures
            if self.with_cl:
                self.cl_update()

            else:
                self.cy_update()

            # reset the disty bit of all particle arrays
            for i in range(self.narrays):
                pa = self.arrays[i]
                pa.set_dirty(False)

            self.is_dirty = False

    def setup_cl(self, context=None):
        """ OpenCL setup for the LinkedListManager  """

        if not context:
            self.context = context = create_some_context()
            self.queue = queue = cl.CommandQueue(context)            
        else:
            self.context = context
            self.queue = queue = cl.CommandQueue(context)

        # allocate the particle array device buffers

        for i in range(self.narrays):
            pa = self.arrays[i]
            pa.setup_cl(context, queue)

        # create the program
        self.setup_program()

    def setup_program(self):
        """ Read the OpenCL kernel source file and build the program """
        src_file = get_pysph_root() + '/base/linked_list.cl'
        src = cl_read(src_file, precision=self.cl_precision)
        self.prog = cl.Program(self.context, src).build()

    def enqueue_copy(self):
        """ Copy the Buffer contents to the host

        The buffers copied are

        cellids, head, next, dix, diy, diz

        """

        if self.with_cl:
        
            for pa in self.arrays:
                enqueue_copy(self.queue, dst=self.cellids[pa.name],
                             src=self.dcellids[pa.name])

                enqueue_copy(self.queue, dst=self.head[pa.name],
                             src=self.dhead[pa.name])
        
                enqueue_copy(self.queue, dst=self.next[pa.name],
                             src=self.dnext[pa.name])
        
                enqueue_copy(self.queue, dst=self.ix[pa.name],
                             src=self.dix[pa.name])
        
                enqueue_copy(self.queue, dst=self.iy[pa.name],
                             src=self.diy[pa.name])
        
                enqueue_copy(self.queue, dst=self.iz[pa.name],
                             src=self.diz[pa.name])
                
