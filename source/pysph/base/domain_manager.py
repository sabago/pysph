import numpy

from pysph.solver.cl_utils import cl_read, get_real, HAS_CL, get_pysph_root,\
    create_some_context, enqueue_copy

import pysph.solver.cl_utils as clu

if HAS_CL:
    import pyopencl as cl
    mf = cl.mem_flags

# Cython functions for neighbor list construction
from linked_list_functions import cbin, unflatten

from point import Point
from cell import py_find_cell_id

# radix sort class
from radix_sort import AMDRadixSort

class DomainManagerType:
    DomainManager = 0
    LinkedListManager = 1
    RadixSortManager = 2

class DomainManager:
    def __init__(self, arrays, context=None, with_cl=True):

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

        # setup OpenCL
        if with_cl:
            if HAS_CL:
                self.with_cl = True
                self._setup_cl(context)
            else:
                raise RuntimeError("PyOpenCL not found!")
        else:
            self.with_cl = False

    #######################################################################
    # public interface
    #######################################################################
    def update(self):
        pass

    #######################################################################
    # object interface
    #######################################################################
    def __iter__(self):
        """The Domain manager produces an iterator for all it's data.
        
        This is needed as the function that will ask for cell
        neighbors should be agnostic about the DomainManager type and
        simply requires a means to iterate through it's data.

        """
        return self

    def next(self):
        raise RuntimeError("Do not iterate over the DomainManager base class!")

    ###########################################################################
    # non-public interface
    ###########################################################################
    def _setup_cl(self, context=None):
        """ OpenCL setup for the CLNNPSManager  """

        if not context:
            self.context = context = create_some_context()
        else:
            self.context = context

        self.queue = queue = cl.CommandQueue(context)

        # allocate the particle array device buffers
        for i in range(self.narrays):
            pa = self.arrays[i]
            pa.setup_cl(context, queue)

        # create the program
        self._setup_program()
    
    def _find_bounds(self):
        """ Find the bounds for the particle arrays.

        The bounds calculated are the simulation cube, defined by the
        minimum and maximum extents of the particle arrays and the
        maximum smoothing length which is used for determining an safe
        cell size for binning.

        """

        inf = numpy.inf

        mx, my, mz = inf, inf, inf
        Mx, My, Mz = -inf, -inf, -inf
        Mh = 0.0

        # update the minimum and maximum for the particle arrays
        for pa in self.arrays:
            pa.read_from_buffer()
            pa.update_min_max(props=['x','y','z','h'])

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

        self._set_cell_size()
        self._find_num_cells()

    def _set_cell_size(self):
        """ Set the cell size for binning

        Notes:
        ------

        If the cell size is being chosen based on the particle
        smoothing lengths, we choose a cell size slightly larger than
        $k\timesh$, where $k$ is the maximum scale factor for the SPH
        kernel. Currently we use the size k + 1

        If no bin sie is provided, the default
        value 2*max(h) is used

        """
        if not self.const_cell_size:
            self.cell_size = get_real((self.kernel_scale_factor+1)*self.Mh,
                                      self.cl_precision)
        else:
            self.cell_size = self.const_cell_size

    def _find_num_cells(self):
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

    def _setup_program(self):
        pass

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

    Next : dict
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

        DomainManager.__init__(self, arrays, context, with_cl)

        # set the kernel scale factor
        self.kernel_scale_factor = kernel_scale_factor

        # set the cell size
        self.const_cell_size = cell_size
        if cell_size:
            self.const_cell_size = get_real(cell_size, self.cl_precision)

        # find global bounds (simulation box and ncells)
        self._find_bounds()

        # The linked list structures for the arrays.
        self.Next = {}
        self.head = {}
        self.cellids = {}
        self.locks = {}

        self.ix = {}
        self.iy = {}
        self.iz = {}

        # device linked list structures
        self.dnext = {}
        self.dhead = {}
        self.dcellids = {}
        self.dlocks = {}

        self.dix = {}
        self.diy = {}
        self.diz = {}

        # dict for kernel launch parameters 
        self.global_sizes = {}
        self.local_sizes = {}

        # initialize counter for the iterator
        self._current_cell = 0

        # initialize the linked list
        self._init_linked_list()

    #######################################################################
    # public interface
    #######################################################################
    def update(self):
        """ Update the linked list """

        # find the bounds for the manager
        self._find_bounds()

        # reset the data structures
        self._init_linked_list()

        # update the data structures
        if self.with_cl:
            self._cl_update()
        else:
            self._cy_update()

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
        
                enqueue_copy(self.queue, dst=self.Next[pa.name],
                             src=self.dnext[pa.name])
        
                enqueue_copy(self.queue, dst=self.ix[pa.name],
                             src=self.dix[pa.name])
        
                enqueue_copy(self.queue, dst=self.iy[pa.name],
                             src=self.diy[pa.name])
        
                enqueue_copy(self.queue, dst=self.iz[pa.name],
                             src=self.diz[pa.name])

    ###########################################################################
    # non-public interface
    ###########################################################################
    def _init_linked_list(self):
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
            self.Next[pa.name] = next
            self.cellids[pa.name] = cellids
            self.locks[pa.name] = locks

            self.ix[pa.name] = ix
            self.iy[pa.name] = iy
            self.iz[pa.name] = iz

        if self.with_cl:
            self._init_device_buffers()

    def _init_device_buffers(self):
        """ Initialize the device buffers """

        for i in range(self.narrays):
            pa = self.arrays[i]
            np = pa.get_number_of_particles()

            # initialize the kerel launch parameters
            self.global_sizes[pa.name] = (np,)
            self.local_sizes[pa.name] = (1,)

            head = self.head[pa.name]
            next = self.Next[pa.name]
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

    def _cy_update(self):
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
                  self.Next[pa.name],
                  mx, my, mz,
                  numpy.int32(ncx), numpy.int32(ncy), numpy.int32(ncz),
                  cell_size, numpy.int32(np),
                  self.mcx, self.mcy, self.mcz
                  )


    def _cl_update(self):
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

    def _setup_program(self):
        """ Read the OpenCL kernel source file and build the program """
        src_file = get_pysph_root() + '/base/linked_list.cl'
        src = cl_read(src_file, precision=self.cl_precision)
        self.prog = cl.Program(self.context, src).build()

    #######################################################################
    # object interface
    #######################################################################
    def next(self):
        """Iterator interface to get cell neighbors.

        Usage:
        ------

            for cell_nbrs in LinkedListManager():
                ...

            where, the length of the iterator is `ncells` and at each call,
            the `forward` neighbors for the cell are returned.


        The `forward` cells for a given cell with index cid are
        neighboring cells with an index cid' >= cid
        
        """
        if self._current_cell == self.ncells:
            self._current_cell = 0
            raise StopIteration
        else:
            # we are getting neighbors for the current cell
            cid = self._current_cell

            # get the cell indices for the current cell to search for
            ncx = self.ncx
            ncy = self.ncy
            ncz = self.ncz

            ix, iy, iz = unflatten(cid, ncx, ncy)

            # determine the range of search
            imin = max(ix -1, 0)
            jmin = max(iy -1, 0)
            kmin = max(iz -1, 0)

            imax = min(ix + 2, ncx)
            jmax = min(iy + 2, ncy)
            kmax = min(iz + 2, ncz)

            # raise the counter for the current cell
            self._current_cell += 1

            return [i+j*ncx+k*ncx*ncy \
                    for i in range(imin, imax) \
                    for j in range(jmin, jmax) \
                    for k in range(kmin, kmax) \
                    if i+j*ncx+k*ncx*ncy >= cid]

    ##########################################################################
    # DEPRECATED
    ##########################################################################
    def reset_cy_data(self):
        for pa in self.arrays:
            head = self.head[pa.name]
            next = self.Next[pa.name]

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

class RadixSortManager(DomainManager):
    """Spatial indexing scheme based on the radix sort.

    The radix sort can be used to determine neighbor information in
    the following way. Consider the particle distribution in an
    idealized one dimensional cell structure as:

    _____________
    |   |   |   |
    | 2 |0,1| 3 |
    |___|___|___|

    that is, particles with indices 0 and 1 are in cell 1, particle 2
    is in cell 0 and 3 in cell 3

    We construct two arrays:

    cellids (size=np) : [1,1,0,2] and
    indices (size=np) : [0,1,2,3]

    and sort the indices based on the keys. After the sorting routine,
    the arrays are:

    cellids (size=np) : [0,1,1,2]
    indices (size=np) : [2,0,1,3]

    Now we can compute an array cell_counts (size=ncells+1) from
    the sorted cellids as:

    cellc = [0, 1, 3, 4],

    which can be computed by launching one thread per particle. If the
    sorted cellid to the left is different from this cellid, then this
    particle is at a cell boundary and the index of that particle in
    the sorted cellids is placed in the `cellc` array at that
    location. Of course, there will be as many cell boundaries as
    there are cells. The boundary conditions will have to be handled
    separately.

    Now using this we can determine the particles that belong to a
    particular cell like so:

    particles in cell0 = indices[ cellids[cellc[0]] : cellids[cellc[1]] ]
    
    """

    def __init__(self, arrays, cell_size=None, context=None,
                 kernel_scale_factor = 2.0, with_cl=True):
        """ Construct a RadixSort manager.

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


        The RadixSort manager constructs and maintains the following
        attributes for each array being indexed:

        (i) cellids (size=np, uint32) : Flattened cell indices for the particles.
        (ii) indices (size=np, uint32) : Particle indices
        (iii) cell_counts(size=ncells+1, uint32) : Cell count array

        The bin size, if provided is constant in each coordinate
        direction. The default choice for the bin size is twice the
        maximum smoothing length for all particles in the domain.

        """

        DomainManager.__init__(self, arrays, context, with_cl)

        # se the kernel scale factor
        self.kernel_scale_factor = kernel_scale_factor

        # set the cell size
        self.const_cell_size = cell_size
        if cell_size is not None:
            self.const_cell_size = get_real(cell_size, self.cl_precision)

        # find global bounds (simulation box and ncells)
        self._find_bounds()

        # The arrays stored for the RadixSortManager
        self.cellids = {}
        self.indices = {}
        self.cell_counts = {}

        # setup the RadixSort objects
        self.rsort = rsort = {}
        self._setup_radix_sort()

        # Corresponding device arrays
        self.dcellids = {}
        self.dindices = {}
        self.dcell_counts = {}

        # dict for kernel launch parameters 
        self.global_sizes = {}
        self.local_sizes = {}

        # initialize counter for the iterator
        self._current_cell = 0

        # initialize the host and device buffers
        self._init_buffers()

    #######################################################################
    # public interface
    #######################################################################
    def update(self):
        """ Update the linked list """

        # find the bounds for the manager
        self._find_bounds()

        # reset the data structures
        self._init_buffers()

        # update the data structures
        if self.with_cl:
            self._cl_update()
        else:
            self._py_update()

    ###########################################################################
    # non-public interface
    ###########################################################################
    def _init_buffers(self):
        """Allocate host and device buffers for the RadixSortManager

        The arrays needed for the manager are:

        (a) cellids of size np which indicates which cell the particle
        belongs to.

        (b) indices of size np which is initially a linear index range
        for the particles. After sorting, this array is used to
        determine particles within a cell.

        (c) cell_counts of size ncells +1 which is used to determine
        the start and end index for the particles within a cell.
        
        """
        # at this point the number of cells is known 
        ncells = self.ncells
        for i in range(self.narrays):
            pa = self.arrays[i]
            
            np = pa.get_number_of_particles()

            # cellids and indices are of length np and dtype uint32
            cellids = numpy.ones(np, numpy.uint32)
            indices = numpy.array(range(np), dtype=numpy.uint32)

            # cell_counts is of length ncells + 1
            cellc = numpy.ones(ncells + 1, numpy.uint32)

            # store these in the dictionary for this particle array
            self.cellids[ pa.name ] = cellids
            self.indices[ pa.name ] = indices
            self.cell_counts[ pa.name ] = cellc

            if self.with_cl:
                self._init_device_buffers()

    def _init_device_buffers(self):
        """Initialize the device buffers

        The arrays initialized here are the cell counts and
        indices. The RadixSort object handles the keys and values.

        """
        narrays = self.narrays
        for i in range(narrays):
            pa = self.arrays[i]

            cellids = self.cellids[pa.name]
            indices = self.indices[pa.name]
            cellc = self.cell_counts[pa.name]

            # Initialize the buffers
            dcellids = cl.Buffer(self.context, mf.READ_WRITE|mf.COPY_HOST_PTR,
                                 hostbuf=cellids)
            
            dindices = cl.Buffer(self.context, mf.READ_WRITE|mf.COPY_HOST_PTR,
                                 hostbuf=indices)
            
            dcellc = cl.Buffer(self.context, mf.READ_WRITE|mf.COPY_HOST_PTR,
                               hostbuf=cellc)

            self.dcellids[pa.name] = cellids
            self.dindices[pa.name] = dindices
            self.dcell_counts[ pa.name ] = dcellc

    def _setup_radix_sort(self):
        """Setup the RadixSort objects to be used.

        Currently, only the AMDRadixSort is available which works on
        both the CPU and the GPU. When the Nvidia's sort is ported,
        we'd have a better option on Nvidia GPUs.

        """
        narrays = self.narrays
        rsort = self.rsort

        if not self.with_cl:
            for pa in self.arrays:
                rsort[pa.name] = AMDRadixSort()

        else:
            ctx = self.context                

            for i in range(narrays):
                pa = self.arrays[i]

                # use the AMD radix sort implementation for CPU devices 
                if clu.iscpucontext(ctx):
                    rsort[ pa.name ] = AMDRadixSort()

                # for Nvidia GPU's we prolly want to use their implemetation
                elif clu.isgpucontext(ctx):
                    raise NotImplementedError
    
    def _cl_update(self):
        """Update the data structures.

        The following three steps are performed in order:

        (a) The particles are binned using a standard algorithm like the one
            for linked lists.

        (b) Sort the resulting cellids (keys) and indices (values) using
            the RadixSort objects

        (c) Compute the cell counts by examining the sorted cellids
        
        """
        # context and queue
        ctx = self.context
        q = self.queue

        # get the cell limits
        ncx, ncy, ncz = self.ncx, self.ncy, self.ncz
        mcx, mcy, mcz = self.mcx, self.mcy, self.mcz
        
        narrays = self.narrays
        for i in range(narrays):
            pa = self.arrays[i]
            np = numpy.uint32( pa.get_number_of_particles() )

            # get launch parameters for this array
            global_sizes = (np,1,1)
            local_sizes = (1,1,1)

            x = pa.get_cl_buffer("x")
            y = pa.get_cl_buffer("y")
            z = pa.get_cl_buffer("z")
            
            # bin the particles to get device cellids
            cellids = self.cellids[pa.name]
            dcellids = self.dcellids[pa.name]
            
            self.prog.bin( q, global_sizes, local_sizes,
                           x, y, z, dcellids, self.cell_size,
                           ncx, ncy, ncz, mcx, mcy, mcz ).wait()
                           
            # read the cellids into host array
            clu.enqueue_copy(q, src=dcellids, dst=cellids)

            # initialize the RadixSort with keys and values
            keys = cellids
            values = self.indices[pa.name]
            
            rsort = self.rsort[ pa.name ]
            rsort.initialize(keys, values, self.context)

            # sort the keys (cellids) and values (indices)
            rsort.sort()
            
            # now compute the cell counts array from the sorted cellids
            # ALSO, DCELLIDS IS PROBABLY PADDED WITH EXTRA ELEMENTS SO WE
            # MUST BE CAREFUL. MAYBE THE GLOBAL SIZE (NP) WILL TAKE CARE
            # OF IT BUT I DON'T KNOW
            dcell_counts = self.dcell_counts[pa.name]
            sortedcellids = rsort.dsortedkeys
            self.prog.compute_cell_counts(q, global_sizes, local_sizes,
                                          sortedcellids, dcell_counts,
                                          self.ncells, np).wait()

            # read the result back to host
            # THIS MAY NEED TO BE DONE OR WE COULD SIMPLY LET IT RESIDE
            # ON THE DEVICE.

    def _py_update(self):
        """Update the data structures using Python"""

        cellsize = self.cell_size
        cellsize1 = 1.0/cellsize

        narrays = self.narrays
        for i in range(narrays):
            pa = self.arrays[i]
            np = pa.get_number_of_particles()

            # bin the particles
            cellids = self.cellids[pa.name]
            x, y, z = pa.get("x", "y", "z")

            for j in range(np):
                _ix = int(numpy.floor( x[j] * cellsize1 ))
                _iy = int(numpy.floor( y[j] * cellsize1 ))
                _iz = int(numpy.floor( z[j] * cellsize1 ))

                cellids[j] = numpy.uint32( (_iz - self.mcz)*self.ncx*self.ncy + \
                                           (_iy - self.mcy)*self.ncx + \
                                           (_ix - self.mcx) )
            
            # sort the cellids and indices
            keys = cellids
            values = self.indices[pa.name]

            rsort = self.rsort[pa.name]
            rsort._sort_cpu(keys, values)

            # compute the cell_count array
            cellc = self.cell_counts[pa.name]
            cellids = keys

            for j in range(np):

                cellid = cellids[j]

                if j == 0:
                    for k in range(cellid + 1):
                        cellc[k] = 0

                elif j == (np - 1):
                    for k in range(cellid+1, self.ncells + 1):
                        cellc[k] = np

                    cellidm = cellids[j-1]
                    for k in range(cellid - cellidm):
                        cellc[cellid - k] = j

                else:
                    cellidm = cellids[j-1]
                    for k in range(cellid - cellidm):
                        cellc[cellid - k] = j
 
    def _setup_program(self):
        """ Read the OpenCL kernel source file and build the program """
        src_file = get_pysph_root() + '/base/radix_sort.cl'
        src = cl_read(src_file, precision=self.cl_precision)
        self.prog = cl.Program(self.context, src).build()
