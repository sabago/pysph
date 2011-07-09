from integrator import Integrator
from cl_utils import HAS_CL, get_pysph_root, get_cl_include,\
     get_scalar_buffer, cl_read, get_real, enqueue_copy

if HAS_CL:
    import pyopencl as cl

from os import path
import numpy

class CLIntegrator(Integrator):

    def setup_integrator(self, context):
        """ Setup the additional particle arrays for integration.

        Parameters:
        -----------

        context -- the OpenCL context

        setup_cl on the calcs must be called when all particle
        properties on the particle array are created. This is
        important as all device buffers will created.

        """
        Integrator.setup_integrator(self)

        self.setup_cl(context)

        self.cl_precision = self.particles.get_cl_precision()

        #self.step_props = ['_tmpx', '_tmpy', '_tmpz']

    def setup_cl(self, context):
        """ OpenCL setup """

        self.context = context
                
        for calc in self.calcs:
            calc.setup_cl(context)

        # setup the OpenCL Program
        root = get_pysph_root()
        src = cl_read(path.join(root, 'solver/integrator.cl'), 
                      self.particles.get_cl_precision())

        self.program = cl.Program(context, src).build(get_cl_include())

    def reset_accelerations(self, step):

        for array in self.arrays:
            queue = array.queue
            np = array.get_number_of_particles()

            to_step = self.step_props[array.name][step]
            for prop in to_step:
                acc_prop = to_step[prop][1]

                acc_buffer = array.get_cl_buffer( acc_prop )
                self.program.set_to_zero(queue, (np,), (1,), acc_buffer).wait()

    def save_initial_arrays(self):
        """ Set the initial arrays for each calc

        The initial array is the update property of a calc appended with _0
        Note that multiple calcs can update the same property and this 
        will not replicate the creation of the initial arrays.

        In OpenCL, we call the EnqueueCopyBuffer with source as the
        current update property and destination as the initial
        property array.
        
        """        
        for array in self.arrays:

            queue = array.queue
            initial_props = self.initial_properties[ array.name ]

            for prop in initial_props:

                src = array.get_cl_buffer( prop )
                dst = array.get_cl_buffer( initial_props[prop] )

                enqueue_copy(queue=queue, src=src, dst=dst)

        # ncalcs = len(calcs)
        # for i in range(ncalcs):
        #     calc = calcs[i]
        #     queue = calc.queue

        #     if calc.integrates:
        #         updates = calc.updates
        #         nupdates = len(updates)

        #         pa = self.arrays[calc.dnum]

        #         for j in range(nupdates):
        #             update_prop = updates[j]
        #             initial_prop = self.initial_props[calc.id][j]

        #             update_prop_buffer = pa.get_cl_buffer(update_prop)
        #             initial_prop_buffer = pa.get_cl_buffer(initial_prop)

        #             enqueue_copy(queue=queue, src=update_prop_buffer,
        #                          dst=initial_prop_buffer)

    # def reset_current_buffers(self, calcs):
    #     """ Reset the current arrays """
        
    #     ncalcs = len(calcs)
    #     for i in range(ncalcs):
    #         calc = calcs[i]
    #         queue = calc.queue

    #         if calc.integrates:

    #             updates = calc.updates
    #             nupdates = len(updates)

    #             pa = self.arrays[calc.dnum]
            
    #             for j in range(nupdates):
    #                 update_prop = updates[j]
    #                 initial_prop = self.initial_props[calc.id][j]

    #                 # get the device buffers
    #                 update_prop_buffer = pa.get_cl_buffer(update_prop)
    #                 initial_prop_buffer = pa.get_cl_buffer(initial_prop)

    #                 # reset the current property to the initial array

    #                 enqueue_copy(queue=queue,src=initial_prop_buffer,
    #                              dst=update_prop_buffer)

    def eval(self):
        """ Evaluate each calc and store in the k list if necessary """

        calcs = self.calcs
        ncalcs = len(calcs)
        particles = self.particles
        
        k_num = self.cstep
        for i in range(ncalcs):
            calc = calcs[i]
            queue = calc.queue

            updates = calc.updates
            nupdates = calc.nupdates

            # get the destination particle array for this calc
            
            pa = dest = calc.dest
            
            if calc.integrates:
                calc.sph( *calc.dst_writes[k_num] )

            else:
                calc.sph( *calc.updates )

                #particles.barrier()

                #self.rupdate_list[calc.dnum] = [update_prop]
                
                #particles.update_remote_particle_properties(
                #    self.rupdate_list)

        #ensure that the eval phase is completed for all processes

        particles.barrier()

    def step(self, dt):
        """ Perform stepping for the integrating calcs """

        cl_dt = get_real(dt, self.cl_precision)

        # get the current stage of the integration
        k_num = self.cstep

        for array in self.arrays:

            # get the number of particles
            np = array.get_number_of_particles()

            # get the command queue for the array
            queue = array.queue
            
            # get the mapping for this array and this stage
            to_step = self.step_props[ array.name ][k_num]

            for prop in to_step:

                initial_prop = to_step[ prop ][0]
                step_prop = to_step[ prop ][1]

                prop_buffer = array.get_cl_buffer( prop )
                step_buffer = array.get_cl_buffer( step_prop )
                initial_buffer = array.get_cl_buffer( initial_prop )

                self.program.step_array(queue, (np,1,1), (1,1,1),
                                        initial_buffer, step_buffer,
                                        prop_buffer, cl_dt)
        self.cstep += 1
        
        # for i in range(ncalcs):
        #     calc = calcs[i]
        #     queue = calc.queue

        #     if calc.integrates:
                
        #         updates = calc.updates
        #         nupdates = calc.nupdates

        #         # get the destination particle array for this calc
            
        #         pa = self.arrays[calc.dnum]
        #         np = pa.get_number_of_particles()

        #         for j in range(nupdates):
        #             update_prop = updates[j]
        #             k_prop = self.k_props[calc.id][k_num][j]

        #             current_buffer = pa.get_cl_buffer(update_prop)
        #             step_buffer = pa.get_cl_buffer(k_prop)
        #             tmp_buffer = pa.get_cl_buffer('_tmpx')
                
        #             self.program.step_array(queue, (np,1,1), (1,1,1),
        #                                     current_buffer, step_buffer,
        #                                     tmp_buffer, cl_dt)

        #             enqueue_copy(queue, src=tmp_buffer,
        #                          dest=current_buffer)
                    
        #         pass
        #     pass

        # # Increment the step by 1

        # self.cstep += 1

##############################################################################
#`CLEulerIntegrator` class 
##############################################################################
class CLEulerIntegrator(CLIntegrator):
    """ Euler integration of the system X' = F(X) with the formula:
    
    X(t + h) = X + h*F(X)
    
    """    
    def __init__(self, particles, calcs):
        CLIntegrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def integrate(self, dt):
        
        # set the initial buffers
        self.save_initial_arrays()

        # Euler step
        self.reset_accelerations(step=1)

        self.eval()
        self.step(dt)

        self.particles.update()

        self.cstep = 1

##############################################################################
#`CLRK2Integrator` class 
##############################################################################
class CLRK2Integrator(CLIntegrator):
    """ RK2 Integration for the system X' = F(X) with the formula:

    # Stage 1
    K1 = F(X)
    X(t + h/2) = X0 + h/2*K1

    # Stage 2
    K1 = F( X(t+h/2) )
    X(t + h) = X0 + h * K1

    """    
    def __init__(self, particles, calcs):
        CLIntegrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def integrate(self, dt):

        # set the initial arrays
        self.save_initial_arrays()  # X0 = X(t)
        
        #############################################################
        # Stage 1
        #############################################################
        self.reset_accelerations(step=1)

        self.eval()                # K1 = F(X)
        self.step(dt/2)            # F(X+h/2) = X0 + h/2*K1

        self.particles.update()

        self.cstep = 1

        #############################################################
        # Stage 2
        #############################################################
        self.reset_accelerations(step=1)

        self.eval()                # K1 = F( X(t+h/2) )
        self.step(dt)              # F(X+h) = X0 + h*K1

        self.particles.update()

        self.cstep = 1

##############################################################################
#`CLPredictorCorrectorIntegrator` class 
##############################################################################
class CLPredictorCorrectorIntegrator(CLIntegrator):
    """ Predictor Corrector Integration of a system X' = F(X) using the scheme
    
    Predict:
    X(t + h/2) = X0 + h/2 * F(X)

    Correct:    
    X(t + h/2) = X0 + h/2 * F( X(t + h/2) )

    Step:
    X(t + h) = 2*X(t + h/2) - X0

    """
    def __init__(self, particles, calcs):
        CLIntegrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def final_step(self):

        for array in self.arrays:

            to_step = self.step_props[array.name][1]
            for prop in to_step:

                current_buffer = array.get_cl_buffer( prop )
                initial_buffer = array.get_cl_buffer( to_step[prop][0] )

                self.program.pc_final_step( queue, (np,), (1,),
                                            current_buffer,
                                            initial_buffer).wait()

    def integrate(self, dt):

        # save the initial arrays
        self.save_initial_arrays()    # X0 = X(t)

        ############################################################
        # Predict
        ############################################################
        self.reset_accelerations(step=1)

        self.eval()                  # K1 = F(X)
        self.step(dt/2)              # X(t+h/2) = X0 + h/2*K1

        self.particles.update()

        self.cstep = 1

        ##############################################################
        # Correct
        ##############################################################
        self.reset_accelerations(step=1)

        self.eval()                  # K1 = F( X(t+h/2) )
        self.step(dt/2)              # X(t+h/2) = X0 + h/2*K1

        self.particles.update()

        ##############################################################
        # Step
        ##############################################################
        self.final_step(dt)           # X(t+h) = 2*X(t+h/2) - X0
        self.particles.update()

        self.cstep = 1
