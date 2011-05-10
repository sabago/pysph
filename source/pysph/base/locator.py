import linked_list_functions as ll
import numpy

class LinkedListSPHNeighborLocator:

    def __init__(self, manager, source, dest, scale_fac=2):
        self.manager = manager
        self.source = source
        self.dest = dest
        self.scale_fac = scale_fac
        
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
        
