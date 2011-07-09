class ParticleType:
    """
    An empty class to provide an enum for the different particle types
    used in PySPH.

    The types defined are:

    Fluid -- The default particle type used to represent fluids.

    Solid -- Use this to represent solids

    DummyFluid --

    Probe --

    Boundary -- Boundary particles that contribute to forces but
    inherit properties from other particles. Use this to avoid
    particle deficiency near boundaries.

    """
    Fluid = 0
    Solid = 1
    DummyFluid = 2
    Probe = 3
    Boundary = 4
    
    def __init__(self):
        """
        Constructor.

        We do not allow this class to be instantiated. Only the class attributes
        are directly accessed. Instantiation will raise an error.
        """
        raise SystemError, 'Do not instantiate the ParticleType class'



    
