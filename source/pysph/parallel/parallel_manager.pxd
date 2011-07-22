# Definition for a ParallelManager

cdef class ParallelManager:
    cdef public object particles

    cpdef initialize(self, particles)

    cpdef update(self)

    cpdef update_remote_particle_properties(self, list props=*)


