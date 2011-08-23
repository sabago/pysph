from pysph.sph.sph_calc import SPHCalc
from pysph.sph.funcs.arithmetic_funcs import PropertyGet

import numpy
import logging
logger = logging.getLogger()

#############################################################################
#`Integrator` class
#############################################################################
class Integrator(object):
    """ The base class for all integrators. Currently, the following 
    integrators are supported:
    
    (a) Forward Euler Integrator
    (b) RK2 Integrator
    (c) RK4 Integrator
    (d) Predictor Corrector Integrator
    (e) Leap Frog Integrator

    The integrator operates on a list of SPHCalc objects which define
    the interaction between a single destination particle array and a
    list of source particle arrays. 

    An instance of SPHCalc is called a `calc` and thus, the integrator
    operates on a list of calcs. The calcs serve as the functions to
    be evaluated for the integrator.

    A calc can be integrating or non integrating depending on the
    operation it represents. For example, the summation density and
    density rate operations result in calcs that are non integrating
    and integrating respectively. Note that both of them operate on
    the same LHS variable, namely the density.

    Example:
    =========

    Consider a dam break simulation with two particle arrays, fluid
    and boundary. The operations are

    (a) Tait equation (updates=['p','cs'])
    (b) Density Rate (updates=['rho'])
    (c) Momentum equation with avisc  (updates = ['u','v'])
    (d) Gravity force (updates = ['u','v'])
    (e) Position Stepping (updates=['x','y'])
    (f) XSPH Correction (updates=['x','y'])


    Integration of this system relies on the use of two dictionaries:

    (1) initial_properties

    The initial_properties for the integrator would look like:

    {
    'fluid': {'x':'_x0', 'y':'_y0', 'u':'_u0', 'v':'_v0', ...},
    'boundary':{'rho':'_rho0'}
    }

    that is, the initial_properties serves as a mapping between names
    of particle properties that need to be stepped and their initial
    values, per particle array. This is needed for multi-step
    integrators since we need the final step is with respect to the
    initial properties. The initial_properties is used to save out the
    properties once at the start of the integration step.

    (2) step_props

    The step_props dictionary looks like:

    {
    'fluid':{1:{ 'x':['_x0', '_a_x_1'], 'rho':['_rho0', '_a_rho_1'] ... }
    'boundary':{1: {'rho':['_rho0', '_a_rho_1']} }
    }

    that is, for each stage of the integration (k1, k2..) a dictionary
    is stored. This dictionary is keyed on the property to be stepped
    and has as value, a list of two strings. The first string is the
    name of the intial array for this property to be stepped and the
    second is the name of the variable in which the acceleration for
    this property is stored.

    The initial_properties and step_props dicts are constructed at
    setup time while examining the calcs for their update
    properties. A single acceleration variable is used for each
    property that needs to be stepped.

    The naming convention for the acceleration variable is
    '_a_<prop_name>_<stage>', thus, the acceleration variable for
    velocity at the 2nd stage of an integrator would be '_a_u_2'

    Using these two dictionaries, a typical step for the integrator is
    the following:

    (a) Save Intial Arrays:
    ------------------------

    This is easily done using the initial_properties dict. A call is
    made to the particle array to copy over the values as represented
    by the mapping.

    (b) Reset Accelerations.
    ------------------------

    Since one acceleration variable is used per property to be
    stepped, the accelerations must be set to zero before the eval
    phase of the integrator. This is because the accelerations will be
    appended at each call to the underlying SPHFuncton.

    (c) Evaluate the RHS.
    -----------------------

    Each calc calls it's eval method with appropriate arguments to
    store the results of the evaluation.

    For integrating calcs, the argument is the acceleration variable
    for that property and that stage of the integration. This is where
    the step_props dict comes in.

    For non integrating calcs, the argument are the update properties
    for that calc. 

    (d) Step
    ---------

    Once the calcs have been evaluated in order, the accelerations are
    stored in the appropriate variables for each particle array.
    Using the step_props dict, we can step the properties for that
    stage.

    (e) Update particles
    ---------------------

    Typically, the positions of the particles will be updated in (d)
    and this means that the indexing scheme is outdated. This
    necessitates an update to recompute the neighbor information.

    """

    def __init__(self, particles=None, calcs=[], pcalcs = []):
        self.particles = particles

        # the calcs used for the RHS evaluations
        self.calcs = calcs

        # the number of steps for the integrator. Typically equal to
        # the number of k arrays required.
        self.nsteps = 1

        # counter for the current stage of the integrator.
        self.cstep = 1

        # global and local time
        self.time = 0.0
        self.local_time = 0.0

        # list of particle properties to be updated across processors.
        self.rupdate_list = []

        # mapping between names of step properties and accelerations
        # per stage, per particle array
        self.step_props = {}

        # mapping between step prop name and it's initial prop name
        # per particle array
        self.initial_properties = {}

        # store the velocity accelerations per array per stage
        self.velocity_accelerations = {}

    def set_rupdate_list(self):
        """ Generate the remote update list.

        The format of this list is tied to ParallelCellManager.

        """
        for i in range(len(self.particles.arrays)):
            self.rupdate_list.append([])

    def setup_integrator( self ):
        """ Setup the integrator.

        This function sets up the initial_properties and step_props
        dicts which are used extensively for the integration.

        A non-integrating calc is used to update the property of some
        variable as a function of other variables ( eg p = f(rho)
        ).

        An integrating calc computes the accelerations for some LHS
        property.

        During the eval phase, a calc must pass in a string defining
        the output arrays to append the RHS result to. For a
        non-integrating calc this is simply the calc's update
        property. For an integrating calc, the arguments must be the
        accelerations for that property.
        """

        # save the arrays for easy reference
        self.arrays = self.particles.arrays

        # intialize the step_props and initial_properties.
        for array in self.arrays:
            self.step_props[array.name] = {}
            self.initial_properties[array.name] = {}

            # Initialize the velocity accelerations dict per array
            self.velocity_accelerations[array.name] = {}

            # step props needs a dict per stage of the integration as well
            for k in range(self.nsteps):
                k_num = k + 1
                self.step_props[array.name][k_num] = {}

                self.velocity_accelerations[array.name][k_num] = {}
        
        for calc in self.calcs:

            # get the destination particle array for the calc
            dest = calc.dest

            updates = calc.updates
            nupdates = len(updates)

            # the initial properties and accelerations need to be
            # defined in the case of integrating calcs
            if calc.integrates:

                for j in range(nupdates):
                    update_prop = updates[j]

                    # define and add the property to the destination array
                    initial_prop = '_' + update_prop + '0'
                    dest.add_property( {"name":initial_prop} )

                    # save the intial property
                    self.initial_properties[dest.name][update_prop]=initial_prop

                    # an acceleration needs to be defined for every stage.
                    for k in range(self.nsteps):
                        k_num = k + 1

                        # define and add the acceleration variable
                        step_prop = '_a_' + update_prop + '_' + str(k_num)
                        dest.add_property( {"name":step_prop} )

                        # save the acceleration variable
                        self.step_props[dest.name][k_num][update_prop] = \
                                                       [initial_prop, step_prop]

                        # tell the calc to use this acceleration
                        # variable as the argument for the eval phase
                        dst_writes = calc.dst_writes.get(k_num)
                        if not dst_writes:
                            calc.dst_writes[k_num] = []

                        calc.dst_writes[k_num].append( step_prop )

        self.set_rupdate_list()

    def reset_accelerations(self, step):
        """ Reset the accelerations.

        Parameters:
        -----------

        step : int
            The stage of the integrator for which to reset the accelerations.
            
        """

        for array in self.arrays:

            zeros = numpy.zeros( array.get_number_of_particles() )
            
            for step_prop in self.step_props[ array.name ][step]:
                acc_prop = self.step_props[array.name][step][step_prop][1]

                array.set(**{acc_prop:zeros} )

    def save_initial_arrays(self):
        """ Save the initial arrays. """
        for array in self.arrays:
            array.copy_over_properties( self.initial_properties[array.name] )

    def eval(self):
        """ Evaluate the LHS as defined by the calcs.

        For evaluations that are time dependant, we rely on the
        itnegrator's local time variable to determine what time we're
        at.

        As an example, an RK2 integrator would perorm two evaluations:

        K1 is evaluated at self.local_time = self.time
        K2 is evaluated at self.local_time = self.time + dt/2

        It is the responsibility of the integrator's `integrate`
        method to update the local time variable used by `eval`

        """

        calcs = self.calcs
        ncalcs = len(calcs)

        particles = self.particles
        
        k_num = self.cstep
        for i in range(ncalcs):
            calc = calcs[i]

            # set the time for the destination particle array
            calc.dest.set_time(self.local_time)

            # Evaluate the calc
            if calc.integrates:

                if calc.tensor_eval:
                    calc.tensor_sph( *calc.dst_writes[k_num] )
                else:
                    calc.sph( *calc.dst_writes[k_num] )

            else:
                calc.sph( *calc.updates )

                # ensure all processes have reached this point
                particles.barrier()

                # update the properties for remote particles
                self.rupdate_list[calc.dnum] = [calc.updates]

                particles.update_remote_particle_properties(
                    self.rupdate_list)                

        # ensure that all processors have evaluated the RHS's
        # not likely that this is necessary.
        particles.barrier()
        
    def step(self, dt):
        """ Step the particle properties. """

        # get the current stage of the integration
        k_num = self.cstep

        for array in self.arrays:

            # get the mapping for this array and this stage
            to_step = self.step_props[ array.name ][k_num]

            for prop in to_step:

                initial_prop = to_step[ prop ][0]
                step_prop = to_step[ prop ][1]

                initial_arr = array.get( initial_prop )
                step_arr = array.get( step_prop )
                updated_array = initial_arr + step_arr * dt

                array.set( **{prop:updated_array} )

                # store the acceleration arrays
                if prop in ['u','v','w']:
                    self.velocity_accelerations[array.name][k_num][step_prop] = step_arr

        # Increment the step by 1
        self.cstep += 1

    def get_max_acceleration(self, array, solver):

        if solver.count == 1:
            return solver.dt
        
        if not ( array in self.arrays ):
            raise RuntimeError("Array %s does not belong to me "%array.name)

        acc = -numpy.inf

        if array.properties.has_key("_a_u_1"):

            dim = solver.dim
            if dim == 1:
                ax = self.step_props[array.name][1]['u'][1]
                k1_x = self.velocity_accelerations[array.name][1][ax]

                acc = max( acc, numpy.max(numpy.abs(k1_x)) )

            elif dim == 2:
                ax = self.step_props[array.name][1]['u'][1]
                k1_x = self.velocity_accelerations[array.name][1][ax]
                
                ay = self.step_props[array.name][1]['v'][1]
                k1_y = self.velocity_accelerations[array.name][1][ay]

                acc = max( acc, numpy.max(numpy.sqrt(k1_x*k1_x +\
                                                     k1_y*k1_y)) )
                                                     
            elif dim == 3:
                ax = self.step_props[array.name][1]['u'][1]
                k1_x = self.velocity_accelerations[array.name][1][ax]
                
                ay = self.step_props[array.name][1]['v'][1]
                k1_y = self.velocity_accelerations[array.name][1][ay]
                
                az = self.step_props[array.name][1]['w'][1]
                k1_z = self.velocity_accelerations[array.name][1][az]
            
                acc = max( acc,
                           numpy.max(numpy.sqrt(k1_x*k1_x + \
                                                k1_y*k1_y + \
                                                k1_z*k1_z)) )
        return acc

    def integrate(self, dt):
        raise NotImplementedError

##############################################################################
#`EulerIntegrator` class 
##############################################################################
class EulerIntegrator(Integrator):
    """ Euler integration of the system X' = F(X) with the formula:
    
    X(t + h) = X + h*F(X)
    
    """    
    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def integrate(self, dt):

        # set the initial properties
        self.save_initial_arrays()      # X0 = X(t)

        # Euler step
        self.reset_accelerations(step=1)

        # set the local time to the integrator's time
        self.local_time = self.time
        self.eval()                    # F(X) = k1
        self.step( dt )                # X(t + h) = X0 + h*k1

        self.particles.update()

        self.cstep = 1

##############################################################################
#`RK2Integrator` class 
##############################################################################
class RK2Integrator(Integrator):
    """ RK2 Integration for the system X' = F(X) with the formula:

    # Stage 1
    K1 = F(X)
    X(t + h/2) = X0 + h/2*K1

    # Stage 2
    K1 = F( X(t+h/2) )
    X(t + h) = X0 + h * K1

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def integrate(self, dt):

        # set the initial arrays
        self.save_initial_arrays()  # X0 = X(t)

        #############################################################
        # Stage 1
        #############################################################
        self.reset_accelerations(step=1)

        # set the local time to the integrator's time
        self.local_time = self.time
        self.eval()                # K1 = F(X)
        self.step(dt/2)            # F(X+h/2) = X0 + h/2*K1

        self.particles.update()

        self.cstep = 1

        #############################################################
        # Stage 2
        #############################################################
        self.reset_accelerations(step=1)

        # update the local time
        self.local_time = self.time + dt/2
        self.eval()                # K1 = F( X(t+h/2) )
        self.step(dt)              # F(X+h) = X0 + h*K1

        self.particles.update()

        self.cstep = 1

##############################################################################
#`RK4Integrator` class 
##############################################################################
class RK4Integrator(Integrator):
    """ RK4 Integration of a system X' = F(X) using the scheme
    
    # Stage 1
    K1 = F(X)
    X(t + h/2) = X0 + h/2*K1

    # Stage 2
    K2 = F( X(t + h/2) )
    X(t + h/2) = X0 + h/2*K2

    # Stage 3
    K3 = F( X(t + h/2) )
    X(t + h) = X0 + h*K3

    # Stage 4
    K4 = F( X(t + h) )
    X(t + h) = X0 + h/6 * ( K1 + 2*K2 + 2*K3 + K4 )

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 4

    def final_step(self, dt):
        """ Perform the final step for RK4 integration """

        fac = 1.0/6.0
        for array in self.arrays:

            to_step_k1 = self.step_props[array.name][1]
            to_step_k2 = self.step_props[array.name][2]
            to_step_k3 = self.step_props[array.name][3]
            to_step_k4 = self.step_props[array.name][4]

            for prop in to_step_k1:

                initial_array = array.get( to_step_k1[prop][0] )

                k1_array = array.get( to_step_k1[prop][1] )
                k2_array = array.get( to_step_k2[prop][1] )
                k3_array = array.get( to_step_k3[prop][1] )
                k4_array = array.get( to_step_k4[prop][1] )

                updated_array = initial_array + fac*dt*(k1_array + \
                                                        2*k2_array + \
                                                        2*k3_array + \
                                                        k4_array)

                array.set( **{prop:updated_array} )   

    def integrate(self, dt):

        # save the initial arrays
        self.save_initial_arrays()   # X0 = X(t)

        #############################################################
        # Stage 1
        #############################################################
        self.reset_accelerations(step=1)

        # set the local time to the integrator's time
        self.local_time = self.time
        self.eval()                 # K1 = F(X)
        self.step(dt/2)             # X(t + h/2) = X0 + h/2*K1

        self.particles.update()

        #############################################################
        # Stage 2
        #############################################################
        self.reset_accelerations(step=2)

        # update the local time
        self.local_time = self.time + dt/2
        self.eval()                 # K2 = F( X(t+h/2) )
        self.step(dt/2)             # X(t+h/2) = X0 + h/2*K2

        self.particles.update()

        #############################################################
        # Stage 3
        #############################################################
        self.reset_accelerations(step=3)

        # update the local time
        self.local_time = self.time + dt/2
        self.eval()                 # K3 = F( X(t+h/2) )
        self.step(dt)               # X(t+h) = X0 + h*K3
        
        self.particles.update()

        #############################################################
        # Stage 4
        #############################################################
        self.reset_accelerations(step=4)

        # update the local_time
        self.local_time = self.time + dt
        self.eval()                 # K4 = F( X(t+h) )
        self.final_step(dt)          # X(t + h) = X0 + h/6(K1 + 2K2 + 2K3 + K4)

        self.particles.update()

        # reset the step counter
        self.cstep = 1

##############################################################################
#`PredictorCorrectorIntegrator` class 
##############################################################################
class PredictorCorrectorIntegrator(Integrator):
    """ Predictor Corrector Integration of a system X' = F(X) using the scheme
    
    Predict:
    X(t + h/2) = X0 + h/2 * F(X)

    Correct:    
    X(t + h/2) = X0 + h/2 * F( X(t + h/2) )

    Step:
    X(t + h) = 2*X(t + h/2) - X0

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def final_step(self):
        """ Perform the final step in the PC integration method """

        for array in self.arrays:

            to_step = self.step_props[array.name][1]
            for prop in to_step:

                current_array = array.get( prop )
                initial_array = array.get( to_step[prop][0] )

                updated_array = 2*current_array - initial_array
                array.set( **{prop:updated_array} )
                              
    def integrate(self, dt):

        # save the initial arrays
        self.save_initial_arrays()    # X0 = X(t)

        ############################################################
        # Predict
        ############################################################
        self.reset_accelerations(step=1)

        # set the local time to the integrator's time
        self.local_time = self.time
        self.eval()                  # K1 = F(X)
        self.step(dt/2)              # X(t+h/2) = X0 + h/2*K1

        self.particles.update()

        self.cstep = 1

        ##############################################################
        # Correct
        ##############################################################
        self.reset_accelerations(step=1)

        # udpate the local time
        self.local_time = self.time + dt/2
        self.eval()                  # K1 = F( X(t+h/2) )
        self.step(dt/2)              # X(t+h/2) = X0 + h/2*K1

        self.particles.update()

        ##############################################################
        # Step
        ##############################################################
        self.final_step()           # X(t+h) = 2*X(t+h/2) - X0
        self.particles.update()

        self.cstep = 1

##############################################################################
#`LeapFrogIntegrator` class 
##############################################################################
class LeapFrogIntegrator(Integrator):
    """ Leap frog integration of a system :
    
    \frac{Dv}{Dt} = F
    \frac{Dr}{Dt} = v
    \frac{D\rho}{Dt} = D
    
    the prediction step:
    
    vbar = v_0 + h * F_0
    r = r_0 + h*v_0 + 0.5 * h * h * F_0
    rhobar = rho_0 + h * D_0

    correction step:
    v = vbar + 0.5*h*(F - F_0)
    
    rho = rhobar + 0.5*h*(D - D_0)

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 2

    def add_correction_for_position(self, dt):
        ncalcs = len(self.icalcs)

        pos_calc = self.pcalcs[0]

        pos_calc_pa = self.arrays[pos_calc.dnum]
        pos_calc_updates = pos_calc.updates
        
        for calc in self.icalcs:

            if calc.tag == "velocity":
                
                pa = calc.dest
                
                updates = calc.updates
                for j in range(calc.nupdates):
                    
                    update_prop = pos_calc_updates[j]

                    #k1_prop = self.k1_props['k1'][calc.id][j]
                    k1_prop = self.k_props[calc.id]['k1'][j]

                    # the current position

                    current_arr = pa.get(update_prop)

                    step_array = pa.get(k1_prop)

                    updated_array = current_arr + 0.5*dt*dt*step_array
                    pos_calc_pa.set(**{update_prop:updated_array})

    def final_step(self, calc, dt):
        #pa = self.arrays[calc.dnum]
        pa = calc.dest
        updates = calc.updates

        for j in range(len(updates)):
            update_prop = updates[j]

            k1_prop = self.k_props[calc.id]['k1'][j]
            k2_prop = self.k_props[calc.id]['k2'][j]

            k1_array = pa.get(k1_prop)
            k2_array = pa.get(k2_prop)

            current_array = pa.get(update_prop)

            updated_array = current_array + 0.5*dt*(k2_array - k1_array)
            
            pa.set(**{update_prop:updated_array})

    def integrate(self, dt):
        
        # set the initial arrays
        
        self.set_initial_arrays()

        # eval and step the non position calcs at the current state
        
        self.do_step(self.ncalcs, dt)

        self.cstep = 1

        # eval and step the position calcs
        
        self.do_step(self.pcalcs, dt)

        # add correction for the positions

        self.add_correction_for_position(dt)

        #for calc in self.hcalcs:
        #    calc.sph('h')

        # ensure all processors have reached this point, then update

        self.particles.barrier()
        self.particles.update()

        # eval and step the non position calcs

        self.eval(self.ncalcs)

        for calc in self.icalcs:
            self.final_step(calc, dt)

        self.cstep = 1


##############################################################################
#`GSPHIntegrator` class 
##############################################################################
class GSPHIntegrator(EulerIntegrator):
    """ Euler integration of the system X' = F(X) with the formula:
    
    X(t + h) = X + h*F(X)
    
    """    

    def step(self, dt):
        """ Step the particle properties. """
        # get the current stage of the integration
        k_num = self.cstep

        for array in self.arrays:

            # get the mapping for this array and this stage
            to_step = self.step_props[ array.name ][k_num]

            for prop in to_step:

                initial_prop = to_step[ prop ][0]
                step_prop = to_step[ prop ][1]

                initial_arr = array.get( initial_prop )
                step_arr = array.get( step_prop )
                updated_array = initial_arr + step_arr * dt

                array.set( **{prop:updated_array} )

                # store the acceleration arrays
                if prop in ['u','v','w']:
                    self.velocity_accelerations[array.name][k_num][step_prop] = step_arr
                    vstar = prop + "star"
                    star = array.get(vstar)
                    star = initial_arr + 0.5 * step_arr*dt
                    array.set( **{vstar:star})

        # Increment the step by 1
        self.cstep += 1

###########################################################################        
    
integration_methods = [('Euler', EulerIntegrator),
                       ('LeapFrog', LeapFrogIntegrator),
                       ('RK2', RK2Integrator),
                       ('RK4', RK4Integrator),
                       ('PredictorCorrector', PredictorCorrectorIntegrator),
                       ]

