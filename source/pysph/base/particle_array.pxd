cimport numpy as np
from pysph.base.carray cimport LongArray, BaseArray


# ParticleTag
# Declares various tags for particles, and functions to check them.

# Note that these tags are the ones set in the 'tag' property of the
# particles, in a particle array. To define additional discrete properties,
# one can add another integer property to the particles in the particle array
# while creating them.

# These tags could be considered as 'system tags' used internally to
# distinguish among different kinds of particles. If more tags are needed for
# a particular application, add them as mentioned above.

# The is_* functions defined below are to be used in Python for tests
# etc. Cython modules can directly use the enum name.

cdef enum ParticleTag:
    LocalReal = 0
    Dummy
    LocalDummy
    RemoteReal
    RemoteDummy
    GhostParticle

cpdef bint is_local_real(long tag)
cpdef bint is_local_dummy(long tag)
cpdef bint is_remote_real(long tag)
cpdef bint is_remote_dummy(long tag)

cpdef long get_local_real_tag()
cpdef long get_dummy_tag()
cpdef long get_local_dummy_tag()
cpdef long get_remote_real_tag()
cpdef long get_remote_dummy_tag()


cdef class ParticleArray:
    """
    Maintains various properties for particles.
    """
    #the type of particles
    cdef public int particle_type

    # dictionary to hold the properties held per particle.
    cdef public dict properties
    cdef public list property_arrays

    # default value associated with each property
    cdef public dict default_values

    # dictionary to hold temporary arrays - we can do away with this.
    cdef public dict temporary_arrays

    # name associated with this particle array
    cdef public str name

    # indicates if coordinates of particles has changed.
    cdef public bint is_dirty
    
    # indicate if the particle configuration has changed.
    cdef public bint indices_invalid

    # the number of real particles.
    cdef public long num_real_particles

    # dictionary to hold the OpenCL properties for a particle
    cdef public dict cl_properties

    # bool indicating CL is setup
    cdef public bint cl_setup_done

    # The OpenCL CommandQueue Context and Device
    cdef public object queue
    cdef public object context
    cdef public object device

    # The OpenCL ParticleArray host and device buffers
    cdef public object pa_buf_host
    cdef public object pa_tag_host

    cdef public object pa_buf_device
    cdef public object pa_tag_device

    cdef object _create_c_array_from_npy_array(self, np.ndarray arr)
    cdef _check_property(self, str)

    cdef np.ndarray _get_real_particle_prop(self, str prop)

    cpdef set_name(self, str name)
    cpdef set_particle_type(self, int particle_type)
    cpdef set_dirty(self, bint val)
    cpdef set_indices_invalid(self, bint val)

    cpdef BaseArray get_carray(self, str prop)

    cpdef int get_number_of_particles(self)
    cpdef remove_particles(self, LongArray index_list)
    cpdef remove_tagged_particles(self, long tag)
    
    # function to add any property
    cpdef add_property(self, dict prop_info)
    cpdef remove_property(self, str prop_name)
    
    # increase the number of particles by num_particles
    cpdef extend(self, int num_particles)

    cpdef has_array(self, str arr_name)

    #function to remove particles with particular value of a flag property.
    cpdef remove_flagged_particles(self, str flag_name, int flag_value)

    # function to get indices of particles that have a particle integer property
    # set to the specified value.
    #cpdef int get_flagged_particles(self, str flag_name, int flag_value,
    # LongArray flag_value)
    
    # get requested properties of selected particles
    # cpdef get_props(self, LongArray indices, *args)

    # set the properties of selected particles int the parray.
    #cpdef int set_particle_props(self, LongArray indices, **props)

    # aligns all the real particles in contiguous positions starting from 0
    cpdef int align_particles(self) except -1

    # add particles from the parray to self.
    cpdef int append_parray(self, ParticleArray parray) except -1

    # create a new particle array with the given particles indices and the
    # properties. 
    cpdef ParticleArray extract_particles(self, LongArray index_array, list
                                          props=*)

    # sets the value of the property flag_name to flag_value for the particles
    # in indices.
    cpdef set_flag(self, str flag_name, int flag_value, LongArray indices)
    cpdef set_tag(self, long tag_value, LongArray indices)

    cpdef copy_properties(self, ParticleArray source, long start_index=*, long
                          end_index=*)


