import numpy

class TimeStep(object):

    def compute_time_step(self, dt):
        return dt

class ViscousTimeStep(TimeStep):

    def __init__(self, cfl, co, particles):
        self.cfl = cfl
        self.co = co

        self.particles = particles

    def compute_time_step(self, dt):

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
    
