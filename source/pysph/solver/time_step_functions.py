import numpy

class TimeStep(object):

    def compute_time_step(self, solver):
        return solver.dt

class ViscousTimeStep(TimeStep):

    def __init__(self, cfl, co, particles):
        self.cfl = cfl
        self.co = co

        self.particles = particles

    def compute_time_step(self, solver):

        cfl = self.cfl
        co = self.co

        # take dt to be some large value
        dt = 1

        arrays = self.particles.arrays
        for array in arrays:
            if array.properties.has_key('dt_fac'):
                        
                h, dt_fac = array.get('h','dt_fac')
                _dt = numpy.min( cfl * h/(co + numpy.max(dt_fac)) )

                # choose the minimum time step from all arrays
                dt = min( _dt, dt )

        return dt

class ViscousAndForceBasedTimeStep(ViscousTimeStep):
    
    def compute_time_step(self, solver):

        # compute the time step based on the viscous criterion
        dt = ViscousTimeStep.compute_time_step(self, solver)

        # compute the acceleration based time step
        integrator = solver.integrator
        arrays = self.particles.arrays

        for array in arrays:

            if array.properties.has_key("_a_u_1"):
                fmax = integrator.get_max_acceleration(array, solver)

                h = array.get("h")
                _dt = self.cfl * numpy.min( numpy.sqrt(h/fmax) )

                dt = min( dt, _dt )

        return dt

class VelocityBasedTimeStep(object):
    def __init__(self, particles, cfl=0.3,):
        self.cfl = cfl
        self.particles = particles

    def compute_time_step(self, solver):
        v = float('inf')
        for pa in solver.particles.arrays:
            val = min(pa.h/(pa.cs+(pa.u**2+pa.v**2+pa.w**2)**0.5))
            if val < v:
                v = val

        return self.cfl*v
