cdef class ParticleArrayData:

    def __init__(self, ParticleArray pa):

        # the particle array supplying the data
        self.pa = pa

        # default arrays
        self.x = pa.get_carray("x")
        self.y = pa.get_carray("y")
        self.z = pa.get_carray("z")

        self.u = pa.get_carray("u")
        self.v = pa.get_carray("v")
        self.w = pa.get_carray("w")

        self.h = pa.get_carray("h")
        self.m = pa.get_carray("m")
        self.rho = pa.get_carray("rho")

        self.p = pa.get_carray("p")
        self.e = pa.get_carray("e")
        self.cs = pa.get_carray("cs")
