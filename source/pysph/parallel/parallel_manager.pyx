
cdef class ParallelManager:
    """Merely defines the standard interface."""

    #cdef public object particles

    cpdef initialize(self, particles):
        """Initialize the parallel manager with the `Particles`.
        """
        raise NotImplementedError

    cpdef update(self):
        """Update the particles.  
        
        This method should basically take care of distributing paticles after
        they have moved.  It should also take care of any load balancing.
        """
        raise NotImplementedError

    cpdef update_remote_particle_properties(self, list props=None):
        """Update only the remote particle properties.

        This is typically called when particles don't move but only some of
        their properties have changed.
        """
        raise NotImplementedError
