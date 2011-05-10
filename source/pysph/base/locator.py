import linked_list_functions as ll
import numpy

class LinkedListSPHNeighborLocator:

    def __init__(self, manager, source, dest, scale_fac=2, cache=False):
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

    def neighbor_loop_code_start(self):
        """ Return a string for the start of the neighbor loop code """

        return """
          int idx = cix[dest_id];
          int idy = ciy[dest_id];
          int idz = ciz[dest_id];

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

    def get_kernel_args(self):
        """ Add the kernel arguments for the OpenCL template """

        dst = self.dest
        src = self.src

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
        
    def get_neighbors(self, i, radius):

        manager = self.manager
        src = self.source
        dst = self.dest

        # construct the bin structure
        manager.find_num_cells()
        manager.update()
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

        nbrs =  ll.get_neighbors(cellid, ix, iy, iz,
                                 ncx, ncy, ncells, head, next)

        x = dst.x.astype(numpy.float32)
        y = dst.y.astype(numpy.float32)
        z = dst.z.astype(numpy.float32)
        return ll.get_neighbors_within_radius(i, radius, x, y, z,
                                              nbrs)
        
