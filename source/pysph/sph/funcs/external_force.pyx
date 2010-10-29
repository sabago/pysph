import numpy

#############################################################################
# `GravityForce` class.
#############################################################################
cdef class GravityForce(SPHFunctionParticle):
    """ Class to compute the boundary force for a fluid boundary pair """ 

    #Defined in the .pxd file
    #cded double gx, gy, gz

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double gx = 0.0, 
                 double gy = 0.0, double gz = 0.0):

        self.id = 'gravityforce'
        self.gx = gx
        self.gy = gy
        self.gz = gz        

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ Perform the boundary force computation """

        nr[0] = self.gx
        nr[1] = self.gy
        nr[2] = self.gz

##############################################################################


#############################################################################
# `Vector` class.
#############################################################################
cdef class VectorForce(SPHFunctionParticle):
    """ Class to compute the boundary force for a fluid boundary pair """ 

    #Defined in the .pxd file
    #cded double fx, fy, fx

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, Point force=Point()):

        self.id = 'gravityforce'
        self.force = force

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ Perform the boundary force computation """

        nr[0] = self.force.x
        nr[1] = self.force.y
        nr[2] = self.force.z

##############################################################################

################################################################################
# `MoveCircleX` class.
################################################################################
cdef class MoveCircleX(SPHFunctionParticle):
    """ Force the x coordinate of a particle to move on a circle.  """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 *args, **kwargs):
        """ Constructor """

        self.id = 'circlex'
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        i = Point(1)
        
        angle = numpy.arccos(i.dot(self._dst)/self._dst.length())

        fx = -numpy.sin(angle)
        
        if self._dst.y < 0:
            fx *= -1

        nr[0] = fx
        nr[1] = 0

###########################################################################

################################################################################
# `MoveCircleY` class.
################################################################################
cdef class MoveCircleY(SPHFunctionParticle):
    """ Force the x coordinate of a particle to move on a circle.  """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 *args, **kwargs):
        """ Constructor """

        self.id = 'circley'
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        i = Point(1)
        
        angle = numpy.arccos(i.dot(self._dst)/self._dst.length())

        fy = numpy.cos(angle)
        
        nr[0] = 0
        nr[1] = fy

###########################################################################
