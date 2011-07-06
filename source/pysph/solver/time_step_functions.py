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

        _dt = dt

        arrays = self.particles.arrays
        for array in arrays:
            if array.properties.has_key('dt_fac'):
                        
                dt_fac = array.get('h','dt_fac')
                _dt = numpy.min( cfl * array.h/(co + numpy.max(dt_fac)) )
                
                if (dt < _dt):
                    dt = _dt

        return dt
    
