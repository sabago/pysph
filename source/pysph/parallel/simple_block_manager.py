"""A parallel manager that uses blocks to partition the domain. At
every iteration, the particles are placed in large bins and these bins
are exchanged across processors. 

"""

from parallel_controller import ParallelController
from parallel_manager import ParallelManager
from parallel_cell import share_data

from pysph.base.fast_utils import arange_long
from pysph.base.particle_array import ParticleArray, get_dummy_tag
from pysph.base.cell import py_construct_immediate_neighbor_list
from pysph.base.cell import CellManager

import numpy

# logger imports
import logging
logger = logging.getLogger()

class ProcessorMap(object):
    """The ProcessorMap determines neighboring processors and a list
    of cells to send to each processor.

    The main data used by the ProcessorMap is the `cells_dict`
    corresponding to each processor's local binning. The cell
    information is used to construct three dictionaries:

    local_cell_map : A dictionary keyed on cell id and with the value
    equal to the local processor rank that created this cell.

    global_cell_map : A dictionary keyed on cell id and with value a
    set of processor ranks that created this cell.

    Two processors may own the same region in space and no attempt is
    made to resolve this conflict. A suitable subclass may provide a
    mechanism to do so.

    """

    def __init__(self, parallel_controller=None):
        """Constructor.

        Parameters:
        -----------

        parallel_controller : pysph.base.parallel.ParallelController
           The controller object which manages the child and
           parent processor ranks required for a global update.

        """
        
        self.parallel_controller = parallel_controller
        if parallel_controller is None:
            self.parallel_controller = ParallelController()

        self.rank = self.parallel_controller.rank
        self.comm = self.parallel_controller.comm

        self.local_cell_map = {}
        self.global_cell_map = {}
        self.conflicts = {}

    def _local_update(self, cells_dict):
        """Update the local cell map.

        The `local_cell_map` is a dictionary keyed on cell id with
        value the rank of te local processor that created this cell.

        """

        self.local_cell_map = {}
        self.global_cell_map = {}

        for cid, cell in cells_dict.iteritems():
            self.local_cell_map[cid] = set( [self.rank] )
            self.global_cell_map[cid] = set( [self.rank] )

    def global_update(self, cells_dict):
        """Update the gglobal cell map.

        The local cell maps from all processors are passed through the
        tree and updated at each stage. After a call to this function,
        every processor has the same gobal cell map.

        The global cell map is keyed on cell id with value, a list of
        processor ranks that created this cell.

        """

        self._local_update(cells_dict)
        self.conflicts = {}

        pc = self.parallel_controller
        comm = self.comm

        # merge data from all children proc maps.
        for c_rank in pc.children_proc_ranks:
            c_cell_map = comm.recv(source=c_rank)

            # merge the data
            for cid in c_cell_map:
                if cid in self.global_cell_map:
                    self.global_cell_map[cid].update( c_cell_map[cid] )
                else:
                    self.global_cell_map[cid] = c_cell_map[cid]

        # we now have partially merged data, send it to parent if not root.
        if pc.parent_rank > -1:
            comm.send(self.global_cell_map, dest=pc.parent_rank)

            # receive updated proc map from parent
            updated_cell_map = comm.recv(source=pc.parent_rank)

            # update the global cell map
            self.global_cell_map.clear()
            self.global_cell_map.update( updated_cell_map )

        # send updated data to children.
        for c_rank in pc.children_proc_ranks:
            comm.send(self.global_cell_map, dest=c_rank)

    def get_cell_list_to_send(self):
        """Return a list of cells to send to each processor.

        Neighboring cells are determined allowing for cells to be
        shared across processors. The return value is a dictionary
        keyed on processor id with value equal to the list of cells to
        send that processor.

        """

        local_map = self.local_cell_map
        global_map = self.global_cell_map
        pc = self.parallel_controller

        cell_list_to_send = {}

        for cid in local_map:

            neighbor_ids = []
            py_construct_immediate_neighbor_list(cid, neighbor_ids,
                                                 include_self=False)

            # handle non-overlapping regions
            for neighbor_id in neighbor_ids:
                if neighbor_id in global_map:
                    owning_pids = list(global_map[neighbor_id])

                    for pid in owning_pids:
                        if not pid in cell_list_to_send:
                            cell_list_to_send[pid] = set([cid])
                        else:
                            cell_list_to_send[pid].update([cid])

            # handle overlapping regions
            conflicting_pids = list(global_map[cid])
            if len(conflicting_pids) > 0:
                for neighbor_id in neighbor_ids:
                    if neighbor_id in local_map:

                        for pid in conflicting_pids:
                            if not pid in cell_list_to_send:
                                cell_list_to_send[pid] = set([cid])
                            else:
                                cell_list_to_send[pid].update([cid])
                    
        return cell_list_to_send            

    def resolve_conflicts(self):
        pass

class SimpleBlockManager(ParallelManager):
    """A parallel manager based on blocks.

    Particles are binned locally with a bin/cell size equal to some
    factor times the maximum smoothing length of the particles. The
    resulting cell structure is used to determine neighboring
    processors using the ProcessorMap and only a single layer of cells
    is communicated.

    """

    def __init__(self, block_scale_factor=6.0):
        """Constructor.

        Parameters:
        -----------

        block_scale_factor : double
            The scale factor to determine the bin size. The smoothing length
            is chosen as: block_scale_factor * glb_max_h

        The block_scale_factor should be greater than or equal to the
        largest kernel radius for all possibly different kernels used
        in a simulation.

        """
        self.parallel_controller = ParallelController()
        self.processor_map = ProcessorMap(self.parallel_controller)
        self.rank = self.parallel_controller.rank

        self.block_scale_factor=block_scale_factor

        self.comm = self.parallel_controller.comm
        self.size = self.parallel_controller.num_procs
        self.rank = self.parallel_controller.rank

        self.glb_bounds_min = [0, 0, 0]
        self.glb_bounds_max = [0, 0, 0]
        self.glb_min_h = 0
        self.glb_max_h = 0

        self.local_bounds_min = [0,0,0]
        self.local_bounds_max = [0,0,0]
        self.local_min_h = 0
        self.local_max_h = 0

        self.local_cell_map = {}
        self.global_cell_map = {}

    ##########################################################################
    # Public interface
    ##########################################################################
    def initialize(self, particles):
        """Initialize the block manager.

        The particle arrays are set and the cell manager is created
        after the cell/block size is computed.

        """
        self.particles = particles
        self.arrays = particles.arrays

        # setup the cell manager
        self._set_dirty()
        self._compute_block_size()
        self._setup_cell_manager()

    def update(self):
        """Parallel update.

        After a call to this function, each processor has it's local
        and remote particles necessary for a simulation.

        """
        cm = self.cm
        pmap = self.processor_map

        # remove all remote particles
        self._remove_remote_particles()

        # bin the particles
        self._rebin_particles()

        # update cell map
        pmap.global_update(cm.cells_dict)

        # set the array pids
        self._set_array_pid()

        # exchange neighbor info
        self._exchange_neighbor_particles()

        # reset the arrays to dirty so locally we are unaffected
        self._set_dirty()

    def update_remote_particle_properties(self, props):
        self.update()

    ###########################################################################
    # Non public interface
    ###########################################################################
    def _add_neighbor_particles(self, data):
        """Append remote particles to the local arrays.

        Parameters:
        -----------

        data : dictionary
            A dictionary keyed on processor id with value equal to a list of
            particle arrays, corresponding to the local arrays in `arrays`
            that contain remote particles from that processor.

        
        """

        arrays = self.arrays
        numarrays = len(arrays)

        remote_particle_indices = []

        for i in range(numarrays):
            num_local = arrays[i].get_number_of_particles()
            remote_particle_indices.append( [num_local, num_local] )

        for pid in data:
            if not pid == self.rank:
                parray_list = data[pid]
                
                for i in range(numarrays):
                    src = parray_list[i]
                    dst = arrays[i]

                    remote_particle_indices[i][1] += src.get_number_of_particles()

                    dst.append_parray(src)

        self.remote_particle_indices = remote_particle_indices
                
    def _get_communication_data(self, cell_list_to_send):
        """Get the particle array data corresponding to the cell list
        that needs to be communicated. """

        numarrays = len(self.arrays)
        cm = self.cm

        data = {}

        for pid, cell_list in cell_list_to_send.iteritems():

            parray_list = []
            for i in range(numarrays):
                parray_list.append(ParticleArray())

            for cid in cell_list:
                cell = cm.cells_dict[cid]
                index_lists = []
                cell.get_particle_ids(index_lists)

                for i in range(numarrays):

                    src = self.arrays[i]
                    dst = parray_list[i]

                    index_array = index_lists[i]

                    pa = src.extract_particles(index_array)

                    # set the local and tag values
                    pa.local[:] = 0
                    pa.tag[:] = get_dummy_tag()

                    dst.append_parray(pa)
                    dst.set_name(src.name)

            data[pid] = parray_list

        return data

        for cid, pids in send_cells_to.iteritems():

            if len(pids) > 0:

                parray_list = []
                
                cell = cm.cells_dict[cid]
                index_lists = []
                cell.get_particle_ids(index_lists)

                for i in range(numarrays):

                    parray_list.append( ParticleArray() )
                    
                    src = self.arrays[i]
                    dst = parray_list[i]

                    index_array = index_lists[i]

                    pa = src.extract_particles(index_array)

                    # set the local and tag values
                    pa.local[:] = 0
                    pa.tag[:] = get_dummy_tag()

                    dst.append(pa)
                    dst.set_name(src.name)

                for pid in pids:
                    to_send[pid] = parray_list
        
    def _exchange_neighbor_particles(self):
        """Send the cells to neighboring processors."""

        pc = self.parallel_controller
        pmap = self.processor_map
        cm = self.cm

        # get the list of cells to send per processor from the processor map
        cell_list_to_send = pmap.get_cell_list_to_send()
        self.cell_list_to_send = cell_list_to_send

        # get the actual particle data to send from the cell manager
        data = self._get_communication_data(cell_list_to_send)

        # share the data
        recv = share_data(self.rank, data.keys(), data, pc.comm, multi=True)

        # add the neighbor particles
        self._add_neighbor_particles(recv)    

    def _rebin_particles(self):
        """Locally recompute the cell structure."""
        cm = self.cm

        # set the particle arrays to dirty
        self._set_dirty()

        # compute the block size
        self._compute_block_size()

        # set the cell size and bin
        cm.cell_size = self.block_size
        cm.rebin_particles()

        # remove any empty cells
        cm.delete_empty_cells()

    def _compute_block_size(self):
        """Compute the block size."""
        self._update_global_properties()
        self.block_size = self.block_scale_factor*self.glb_max_h

    def _setup_cell_manager(self):
        """Set the cell manager used for binning."""
        self.cm = CellManager(arrays_to_bin=self.arrays,
                              min_cell_size=self.block_size,
                              max_cell_size=self.block_size,
                              initialize=True)

    def _set_dirty(self):
        """Set the dirty bit for each particle array."""
        for array in self.arrays:
            array.set_dirty(True)

    def _remove_remote_particles(self):
        """Remove all remote particles."""
        for array in self.arrays:
            to_remove = arange_long(array.num_real_particles,
                                    array.get_number_of_particles())
            array.remove_particles(to_remove)

    def _set_array_pid(self):
        """Set the processor id for each particle array."""
        for array in self.arrays:
            array.set_pid(self.rank)

    def _barrier(self):
        """Wait till all processors reach this point."""
        self.parallel_controller.comm.barrier()

    def _update_global_properties(self):
        """ Exchange bound and smoothing length information among all
        processors.

        Notes:
        ------
        
        At the end of this call, the global min and max values for the 
        coordinates and smoothing lengths are stored in the attributes
        glb_bounds_min/max, glb_min/max_h 

        """
        data_min = {'x':0, 'y':0, 'z':0, 'h':0}
        data_max = {'x':0, 'y':0, 'z':0, 'h':0}
        
        for key in data_min.keys():
            mi, ma = self._find_min_max_of_property(key)
            data_min[key] = mi
            data_max[key] = ma

        self.local_bounds_min[0] = data_min['x']
        self.local_bounds_min[1] = data_min['y']
        self.local_bounds_min[2] = data_min['z']
        self.local_bounds_max[0] = data_max['x']
        self.local_bounds_max[1] = data_max['y']
        self.local_bounds_max[2] = data_max['z']
        self.local_min_h = data_min['h']
        self.local_max_h = data_max['h']

        pc = self.parallel_controller
        
        glb_min, glb_max = pc.get_glb_min_max(data_min, data_max)

        self.glb_bounds_min[0] = glb_min['x']
        self.glb_bounds_min[1] = glb_min['y']
        self.glb_bounds_min[2] = glb_min['z']
        self.glb_bounds_max[0] = glb_max['x']
        self.glb_bounds_max[1] = glb_max['y']
        self.glb_bounds_max[2] = glb_max['z']
        self.glb_min_h = glb_min['h']
        self.glb_max_h = glb_max['h']

        logger.info('(%d) bounds : %s %s'%(pc.rank, self.glb_bounds_min,
                                           self.glb_bounds_max))
        logger.info('(%d) min_h : %f, max_h : %f'%(pc.rank, self.glb_min_h,
                                                   self.glb_max_h))

    def _find_min_max_of_property(self, prop_name):
        """ Find the minimum and maximum of the property among all arrays 
        
        Parameters:
        -----------

        prop_name -- the property name to find the bounds for

        """
        min = 1e20
        max = -1e20

        num_particles = 0
        
        for arr in self.arrays:
            
            if arr.get_number_of_particles() == 0:
                continue
            else:
                num_particles += arr.get_number_of_particles()                

                min_prop = numpy.min(arr.get(prop_name))
                max_prop = numpy.max(arr.get(prop_name))

                if min > min_prop:
                    min = min_prop
                if max < max_prop:
                    max = max_prop

        return min, max
