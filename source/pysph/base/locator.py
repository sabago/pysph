import linked_list_functions as ll
import numpy

# PySPH imports
from carray import LongArray

class OpenCLNeighborLocatorType:
    AllPairNeighborLocator = 0
    LinkedListSPHNeighborLocator = 1
    

class OpenCLNeighborLocator(object):
    pass

class LinkedListSPHNeighborLocator(OpenCLNeighborLocator):

    def __init__(self, manager, source, dest, scale_fac=2.0, cache=False):
        """ Create a neighbor locator between a ParticleArray pair.

        A neighbor locator interfaces with a domain manager which
        provides an indexing scheme for the particles. The locator
        knows how to interpret the information generated after the
        domain manager's `update` function has been called.

        For the locators based on linked lists as the domain manager,
        the head and next arrays are used to determine the neighbors.

        Note:
        -----

        Cython functions to retrieve nearest neighbors given a
        destination particle index is only used when OpenCL support is
        not available.

        When OpenCL is available, the preferred approach is to
        generate the neighbor loop code and kernel arguments and
        inject this into the CL template files (done by CLCalc)

        Parameters:
        -----------

        manager : DomainManager
            The domain manager to use for locating neighbors

        source, dest : ParticleArray
            pair for which neighbors are sought.

        scale_fac : REAL
            Radius scale factor for non OpenCL runs.

        cache : bool
            Flag to indicate if neighbors are to be cached.

        """
        self.manager = manager
        self.source = source
        self.dest = dest

        self.scale_fac = scale_fac
        self.with_cl = manager.with_cl
        self.cache = cache

        # Initialize the cache if using with Cython
        self.particle_cache = []
        if self.cache:
            self.initialize_cache()

        # set the dirty bit to True
        self.is_dirty = True

    def neighbor_loop_code_start(self):
        """ Return a string for the start of the neighbor loop code """

        return """
          int idx = cix[dest_id];
          int idy = ciy[dest_id];
          int idz = ciz[dest_id];

          REAL tmp = ncx*ncy;
          int src_id, cid;

          for (int ix = idx-1; ix <= idx+1; ++ix )
          {
            for (int iy = idy-1; iy <= idy+1; ++iy)
              {
                for (int iz = idz-1; iz <= idz+1; ++iz)
                  {
                    if ( (ix >=0) && (iy >=0) && (iz >= 0) )
                      {
                        cid = (ix) + (iy * ncx) + (iz * tmp);
		  
                        if ( cid < ncells )
                          {
		  
                            src_id = head[ cid ];

                            while ( src_id != -1 )

               """

    def neighbor_loop_code_end(self):
        """ Return a string for the start of the neighbor loop code """

        return """

                    } // if cid < ncells
                            
		 } // if ix >= 0

              } // for iz

	   } // for iy

         } // for ix

         """

    def neighbor_loop_code_break(self):
        return "src_id = next[ src_id ]; "

    def get_kernel_args(self):
        """ Add the kernel arguments for the OpenCL template """

        dst = self.dest
        src = self.source

        cellids = self.manager.dcellids[dst.name]
        cix = self.manager.dix[dst.name]
        ciy = self.manager.diy[dst.name]
        ciz = self.manager.diz[dst.name]
                
        head = self.manager.dhead[src.name]
        next = self.manager.dnext[src.name]
        
        return {'int const ncx': self.manager.ncx,
                'int const ncy': self.manager.ncy,
                'int const ncells': self.manager.ncells,
                '__global uint* cellids': cellids,
                '__global uint* cix': cix,
                '__global uint* ciy': ciy,
                '__global uint* ciz': ciz,
                '__global int* head': head,
                '__global int* next': next
                }
    
    def get_nearest_particles(self, i, output_array, exclude_index=-1):
        """ Return nearest particles from source array to the dest point.

        The search radius is the scale factor times the particle's h

        Parameters:
        -----------

        i : int
            The destination index 

        output_array : LongArray
            Neighbor indices for the point

        exclude_index : int
            Optional index to exclude from the neighbor list
            NOTIMPLEMENTED!
            
        """
        if self.cache:
            return self.neighbor_cache[i]
        else:
            self.get_nearest_particles_nocahe(i, output_array)

    def get_nearest_particles_nocahe(self, i, output_array, exclude_index=-1):

        manager = self.manager
        src = self.source
        dst = self.dest
        
        # Enqueue a copy if the binning is done with OpenCL
        manager.enqueue_copy()

        # get the bin structure parameters
        ncx = manager.ncx
        ncy = manager.ncy
        ncells = manager.ncells

        # linked list for the source
        head = manager.head[src.name]
        next = manager.next[src.name]
        
        # cellid for the destination
        cellid  = manager.cellids[dst.name][i]
        ix = manager.ix[dst.name][i]
        iy = manager.iy[dst.name][i]
        iz = manager.iz[dst.name][i]
        
        # get all neighbors from the 27 neighboring cells
        nbrs =  ll.get_neighbors(cellid, ix, iy, iz,
                                 ncx, ncy, ncells, head, next)
        
        x = dst.x.astype(numpy.float32)
        y = dst.y.astype(numpy.float32)
        z = dst.z.astype(numpy.float32)
        h = dst.h.astype(numpy.float32)

        radius = self.scale_fac * h[i]

        # filter the neighbors to within a cutoff radius
        nbrs = ll.get_neighbors_within_radius(i, radius, x, y, z,
                                              nbrs)
        
        output_array.resize( len(nbrs) )
        output_array.set_data( nbrs )

    def initialize_cache(self):
        """ Iniitialize the particle neighbor cache contents.

        The particle cache is one LongArray for each destination particle.

        """
        np = self.dest.get_number_of_particles()
        self.particle_cache = [ LongArray() for i in range(np) ]

    def update(self):
        """ Update the bin structure and compute cache contents if
        necessary."""

        # update the domain manager
        self.manager.update()

        if self.is_dirty:

            if self.cache:

                self.initialize_cache()

                self._udpdate_cache()

            self.is_dirty = False

    def update_status(self):
        """ Update the dirty bit for the locator and the DomainManager"""
        if not self.is_dirty:
            self.is_dirty = self.source.is_dirty or self.dest.is_dirty

        self.manager.update_status()

    def _udpdate_cache(self):
        """ Compute the contents of the cache """

        np = self.dest.get_number_of_particles()

        for i in range(np):
            nbrs = self.particle_cache[i]

            self.get_nearest_particles_nocahe(i, nbrs)


class AllPairNeighborLocator(OpenCLNeighborLocator):

    def __init__(self, source, dest, scale_fac=2.0, cache=False):
        """ Create a neighbor locator between a ParticleArray pair.

        A neighbor locator interfaces with a domain manager which
        provides an indexing scheme for the particles. The locator
        knows how to interpret the information generated after the
        domain manager's `update` function has been called.

        For the locators based on linked lists as the domain manager,
        the head and next arrays are used to determine the neighbors.

        Note:
        -----

        Cython functions to retrieve nearest neighbors given a
        destination particle index is only used when OpenCL support is
        not available.

        When OpenCL is available, the preferred approach is to
        generate the neighbor loop code and kernel arguments and
        inject this into the CL template files (done by CLCalc)

        Parameters:
        -----------

        source, dest : ParticleArray
            pair for which neighbors are sought.

        scale_fac : REAL
            Radius scale factor for non OpenCL runs.

        cache : bool
            Flag to indicate if neighbors are to be cached.

        """
        self.manager = None
        self.source = source
        self.dest = dest

        self.scale_fac = scale_fac
        self.with_cl = True

        # Explicitly set the cache to false
        self.cache = False

        # Initialize the cache if using with Cython
        self.particle_cache = []

        # set the dirty bit to True
        self.is_dirty = True

    def neighbor_loop_code_start(self):
        """ Return a string for the start of the neighbor loop code """

        return "for (int src_id=0; src_id<nbrs; ++src_id)"

    def neighbor_loop_code_end(self):
        """ Return a string for the start of the neighbor loop code """

        return """ """

    def neighbor_loop_code_break(self):
        return ""

    def get_kernel_args(self):
        """ Add the kernel arguments for the OpenCL template """

        src = self.source
        np = numpy.int32(src.get_number_of_particles())

        return {'int const nbrs': np}

    def update(self):
        """ Update the bin structure and compute cache contents if
        necessary."""

        if self.is_dirty:
            self.is_dirty = False

    def update_status(self):
        """ Update the dirty bit for the locator and the DomainManager"""
        if not self.is_dirty:
            self.is_dirty = self.source.is_dirty or self.dest.is_dirty
