from parallel_manager import ParallelManager
from parallel_controller import ParallelController

from pysph.base.particle_array import get_local_real_tag, get_dummy_tag
from pysph.base.fast_utils import arange_long

# logger imports
import logging
logger = logging.getLogger()

# Constants
Dummy = get_dummy_tag()
LocalReal = get_local_real_tag()


class SimpleParallelManager(ParallelManager):
    """This is a very simple parallel manager.  It simply broadcasts all the
    particles.  Each machine has exactly the same particles for all time.
    There is no support currently for dynamically changing the particles but
    that should be trivial to add.
    """

    def __init__(self, parallel_controller=None):
        if parallel_controller is None:
            parallel_controller = ParallelController()
        self.parallel_controller = parallel_controller

        self.comm = parallel_controller.comm
        self.size = self.parallel_controller.num_procs
        self.rank = self.parallel_controller.rank

    def initialize(self, particles):
        """Initialize the parallel manager with the `Particles`.
        """
        self.particles = particles

    def update(self):
        """Update particles.  This method simply partitions the particles
        equally among the processors.
        """
        logger.debug("SimpleParallelManager.update()")
        comm = self.comm
        rank = self.rank
        size = self.size
        
        local_data = self.particles.arrays

        # Remove remotes from the local.
        for arr in local_data:
            remove = arange_long(arr.num_real_particles, arr.get_number_of_particles())
            arr.remove_particles(remove)

            # everybody sets the pid for their local arrays
            arr.set_pid(rank)

        comm.Barrier()

        # Collect all the local arrays and then broadcast them.
        data = comm.gather(local_data, root=0)
        data = comm.bcast(data, root=0)

        # Now set the remote data's tags to Dummy and add the arrays to
        # the local.
        for i in range(size):
            if i != rank:
                for j, arr in enumerate(data[i]):
                    tag = arr.get_carray('tag')
                    tag.get_npy_array()[:] = Dummy
                    #local = arr.get_carray('local')
                    #local.get_npy_array()[:] = 0

                    local_data[j].append_parray(arr)

        return

    def update_remote_particle_properties(self, props):
        """Update only the remote particle properties.

        This is typically called when particles don't move but only some of
        their properties have changed.
        """
        logger.debug("SimpleParallelManager.update_remote_particle_properties()")
        # Just call update.
        self.update()

