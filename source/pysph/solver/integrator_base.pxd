"""
Contains base classes for all integrators.
"""

# local imports
from pysph.solver.base cimport Base
from pysph.solver.solver_component cimport SolverComponent
from pysph.solver.entity_base cimport EntityBase

# forward declarations.
cdef class Integrator

################################################################################
# `TimeStep` class.
################################################################################
cdef class TimeStep(Base):
    """
    Class to hold the current timestep.

    Making this a separate class makes it easy to reference one copy of it at
    all places needing this value.
    """
    cdef public double time_step

################################################################################
# `ODESteper` class.
################################################################################
cdef class ODESteper(SolverComponent):
    """
    Class to step a given property by a given time step.
    """
    # list of entities whose properties have to be stepped.
    cdef public list entity_list

    # names of properties that have to be stepped
    cdef public list integral_names

    # names of arrays where the values of the next step will be stored.
    cpdef public list next_step_names

    # names of properties representing the 'rate' of change of properties that
    # have to be stepped.
    cdef public list integrand_names

    # the time_step object to obtain the time step.
    cdef public TimeStep time_step
    
    cpdef int setup_component(self) except -1
    cpdef add_entity(self, EntityBase entity)
    cdef int compute(self) except -1

################################################################################
# `Integrator` class.
################################################################################
cdef class Integrator(SolverComponent):
    """
    Base class for all integrators. Integrates a set of given properties.
    """
    # the final list of components that will be executed at every call to
    # compute().
    cdef public list execute_list
    
    # list of entities whose properties have to be integrated.
    cdef public set entity_list

    # the dimension of the velocity and position vectors.
    cpdef public int dimension

    # the time step to use for stepping.
    cpdef public TimeStep curr_time_step

    # add an entity whose properties have to be integrated.
    cpdef add_entity(self, EntityBase entity)

    # add a component to be executed before integration of this property.
    cpdef add_component(self, str property, str comp_name, bint pre_step=*)
    
    # add a component to be exectued before integration of any property is done.
    cpdef add_pre_integration_component(self, str comp_name, bint
                                        at_tail=*)
    
    # set the order in which properties should be integrated.
    cpdef set_integration_order(self, list order)
    
    cdef int compute(self) except -1
    
    # setup the component once prior to execution.
    cpdef int setup_component(self) except -1

    # add a new property to be integrated along with arrays representing the
    # properties. 
    cpdef add_property(self, str prop_name, list integrand_arrays, list
                       integral_arrays, list entity_types=*)

    cpdef set_dimension(self, int dimension)
    
