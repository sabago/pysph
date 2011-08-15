""" Implementations for the basic SPH functions """

from pysph.base.point cimport cPoint, cPoint_new, cPoint_sub, \
     cPoint_distance

from pysph.base.particle_array cimport LocalReal

cdef extern from "math.h":
    double sqrt(double)

################################################################################
# `SPH` class.
################################################################################
cdef class SPH(CSPHFunctionParticle):
    """ Basic SPH Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str prop_name='rho', **kwargs):
        """ Constructor for SPH

        Parameters
        ----------
        source : The source particle array
        dest : The destination particle array

        """
        self.prop_name = prop_name
        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'sph'

        self.cl_kernel_src_file = "basic_funcs.clt"
        self.cl_kernel_function_name = "SPH"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho',self.prop_name] )
        self.dst_reads.extend( ['x','y','z','h','tag'] )

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

        self.src_reads.append(self.prop_name)
        self.dst_reads.append(self.prop_name)

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):

        r""" 
        Perform an SPH interpolation of the property `prop_name` 

        The expression used is:
        
        :math:`$<f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \W_{ab}$`
            
        """
        
        cdef double w, wa, wb,  temp
        cdef double rhob, mb, fb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        rhob = self.s_rho.data[source_pid]
        fb = self.s_prop.data[source_pid]
        mb = self.s_m.data[source_pid]

        h = 0.5*(self.s_h.data[source_pid] + 
                 self.d_h.data[dest_pid])
            
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        if self.hks:
            wa = kernel.function(self._dst, self._src, ha)
            wb = kernel.function(self._dst, self._src, hb)

            w = 0.5 * (wa + wb)

        else:
            w = kernel.function(self._dst, self._src, hab)
        
        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            dnr[0] += w*mb/rhob
        
        nr[0] += w*mb*fb/rhob        

###########################################################################


###############################################################################
# `SPHSimpleGradient` class.
###############################################################################
cdef class SPHSimpleGradient(SPHFunctionParticle):
    """ Basic SPH Derivative Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest,
                 str prop_name='rho',  *args, **kwargs):
        """ Constructor for SPH

        Parameters
        ----------
        source : The source particle array
        dest : The destination particle array
        
        """
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     *args, **kwargs)
        self.id = 'sphd'

        self.cl_kernel_src_file = "basic_funcs.cl"
        self.cl_kernel_function_name = "SPHSimpleGradient"

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho',self.prop_name] )
        self.dst_reads.extend( ['x','y','z','h','tag'] )

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):
        """ 
        Perform an SPH interpolation of the property `prop_name` 

        The expression used is:
        
        """
        cdef double temp
        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        h=0.5*(self.s_h.data[source_pid] + 
               self.d_h.data[dest_pid])
            
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

            # grad.set((grada.x + gradb.x)*0.5,
            #          (grada.y + gradb.y)*0.5,
            #          (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)
        
        temp = self.s_prop[source_pid]
        temp *= self.s_m.data[source_pid]/self.s_rho.data[source_pid]

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)
            
        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z

###########################################################################

################################################################################
# `SPHGrad` class.
################################################################################
cdef class SPHGradient(SPHFunctionParticle):
    """ Basic SPH Gradient Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest,
                 str prop_name='rho',  *args, **kwargs):
        """ Constructor for SPH

        Parameters
        ----------
        source : The source particle array
        dest : The destination particle array

        Notes
        -----
        By default, the arrays are not setup. This lets us set the prop
        name after intialization and then setup the arrays.
        
        """
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     *args, **kwargs)
        self.id = 'sphgrad'

        self.cl_kernel_src_file = "basic_funcs.cl"
        self.cl_kernel_function_name = "SPHGradient"

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho',self.prop_name] )
        self.dst_reads.extend( ['x','y','z','h','tag'] )

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """ Perform an SPH interpolation of the property `prop_name` 

        """
        cdef double temp
        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        h=0.5*(self.s_h.data[source_pid] + 
               self.d_h.data[dest_pid])
    
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

            # grad.set((grada.x + gradb.x)*0.5,
            #          (grada.y + gradb.y)*0.5,
            #          (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)
            
        temp = (self.s_prop[source_pid] -  self.d_prop[dest_pid])
        
        temp *= self.s_m.data[source_pid]/self.s_rho.data[source_pid]
            
        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z

###########################################################################

###############################################################################
# `SPHLaplacian` class.
###############################################################################
cdef class SPHLaplacian(SPHFunctionParticle):
    """ Estimation of the laplacian of a function.  The expression is
     taken from the paper: 

     "Accuracy of SPH viscous flow models",
     David I. Graham and Jason P. Huges, IJNME, 2008, 56, pp 1261-1269.

     """
    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str prop_name='rho',  *args, **kwargs):
        """ Constructor

        Parameters
        ----------
        source : The source particle array
        dest : The destination particle array
        
        """
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     *args, **kwargs)
        self.id = 'sphlaplacian'

        self.cl_kernel_src_file = "basic_funcs.cl"
        self.cl_kernel_function_name = "SPHLaplacian"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m','rho',self.prop_name] )
        self.dst_reads.extend( ['x','y','z','h','tag',self.prop_name] )        

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)
        self.s_prop = self.source.get_carray(self.prop_name)
        self.d_prop = self.dest.get_carray(self.prop_name)

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        r""" 
        Perform an SPH interpolation of the property `prop_name` 

        The expression used is:
        
        :math:`$<f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \\nabla_aW_{ab}$`
        
        """
        cdef double mb, rhob, fb, fa, tmp, dot
        cdef cPoint grad, grada, gradb, rab

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])
        
        mb = self.s_m.data[source_pid]
        rhob = self.s_rho.data[source_pid]
        fb = self.s_prop[source_pid]
        fa = self.d_prop[dest_pid]
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        rab.x = self._dst.x-self._src.x
        rab.y = self._dst.y-self._src.y
        rab.z = self._dst.z-self._src.z
        
        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

            # grad.set((grada.x + gradb.x)*0.5,
            #          (grada.y + gradb.y)*0.5,
            #          (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)
        
        dot = cPoint_dot(rab, grad)
        tmp = 2*mb*(fa-fb)/(rhob*cPoint_length(rab))
        
        nr[0] += tmp*dot

################################################################################
# `CountNeighbors` class.
################################################################################
cdef class CountNeighbors(SPHFunctionParticle):
    """ Count Neighbors.  """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'nbrs'

    def set_src_dst_reads(self):
        pass

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        result[0] += self.nbr_locator.get_nearest_particles(dest_pid).length

################################################################################
# `VelocityGradient3D` class.
################################################################################
cdef class VelocityGradient3D(SPHFunctionParticle):
    r""" Compute the SPH evaluation for the velocity gradient tensor.

    The expression for the velocity gradient is:

    :math:`$\frac{\partial v^i}{\partial x^j} = \sum_{b}\frac{m_b}{\rho_b}(v_b
    - v_a)\frac{\partial W_{ab}}{\partial x_a^j}$`


    The tensor properties are stored in the variables v_ij where 'i'
    refers to the velocity component and 'j' refers to the spatial
    component. Thus :math:`$v_21$` is

    :math:`$\frac{\partial w}{\partial y}$`

    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

        self.id = 'vgrad3D'
        self.tag = "vgrad3D"

        # setup the default properties if they do not exist
        dest_properties = dest.properties.keys()
        tensor_props = ["v_00", "v_01", "v_02",
                        "v_10", "v_11", "v_12",
                        "v_20", "v_21", "v_22"]

        for prop in tensor_props:
            if prop not in dest_properties:
                dest.add_property( dict(name=prop) )

        # set the to_reset variable to tensor_props as we want each of
        # the varaibles to be set to zero at the calc level
        self.to_reset = tensor_props

    def set_src_dst_reads(self):
        pass

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        """ Overide the basic function to call the tensor eval function """
        self.tensor_eval(kernel)

    cpdef tensor_eval(self, KernelBase kernel):

        # get the tag array pointer
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        # perform any iteration specific setup
        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        cdef size_t a

        cdef double result[9]

        cdef DoubleArray v_00 = self.dest.get_carray("v_00")
        cdef DoubleArray v_01 = self.dest.get_carray("v_01")
        cdef DoubleArray v_02 = self.dest.get_carray("v_02")

        cdef DoubleArray v_10 = self.dest.get_carray("v_10")
        cdef DoubleArray v_11 = self.dest.get_carray("v_11")
        cdef DoubleArray v_12 = self.dest.get_carray("v_12")

        cdef DoubleArray v_20 = self.dest.get_carray("v_20")
        cdef DoubleArray v_21 = self.dest.get_carray("v_21")
        cdef DoubleArray v_22 = self.dest.get_carray("v_22")

        for a in range( np ):
            if tag_arr.data[a] == LocalReal:
                self.eval_single(a, kernel, result)

                v_00.data[a] += result[0]
                v_01.data[a] += result[1]
                v_02.data[a] += result[2]
                
                v_10.data[a] += result[3]
                v_11.data[a] += result[4]
                v_12.data[a] += result[5]
                
                v_20.data[a] += result[6]
                v_21.data[a] += result[7]
                v_22.data[a] += result[8]

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        
        cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        cdef size_t nnbrs = nbrs.length
        cdef size_t j

        # the particle itself does not contribute to the gradient
        if self.source is self.dest:
            nnbrs -= 1

        result[0] = result[1] = result[2] = 0.0
        result[3] = result[4] = result[5] = 0.0
        result[6] = result[7] = result[8] = 0.0

        for j in range(nnbrs):
            self.eval_nbr( nbrs.data[j], dest_pid, kernel, result )

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double * result):

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef cPoint vba, grad, grada, gradb
        cdef double tmp

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        vba.x = self.s_u.data[source_pid] - self.d_u.data[dest_pid]
        vba.y = self.s_v.data[source_pid] - self.d_v.data[dest_pid]
        vba.z = self.s_w.data[source_pid] - self.d_w.data[dest_pid]

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = 0.5 * ( grada.x + gradb.x )
            grad.y = 0.5 * ( grada.y + gradb.y )
            grad.z = 0.5 * ( grada.z + gradb.z )

        else:
            grad = kernel.gradient( self._dst, self._src, 0.5*(ha+hb) )

        tmp = mb/rhob

        # the first row is for u,x u,y and u,z
        result[0] += tmp * vba.x * grad.x
        result[1] += tmp * vba.x * grad.y
        result[2] += tmp * vba.x * grad.z

        # the second row is for v,x v,y and v,z
        result[3] += tmp * vba.y * grad.x
        result[4] += tmp * vba.y * grad.y
        result[5] += tmp * vba.y * grad.z

        # the second row is for w,x w,y and w,z
        result[6] += tmp * vba.z * grad.x
        result[7] += tmp * vba.z * grad.y
        result[8] += tmp * vba.z * grad.z

################################################################################
# `VelocityGradient2D` class.
################################################################################
cdef class VelocityGradient2D(SPHFunctionParticle):
    r""" Compute the SPH evaluation for the velocity gradient tensor in 2D.

    The expression for the velocity gradient is:

    :math:`$\frac{\partial v^i}{\partial x^j} = \sum_{b}\frac{m_b}{\rho_b}(v_b
    - v_a)\frac{\partial W_{ab}}{\partial x_a^j}$`


    The tensor properties are stored in the variables v_ij where 'i'
    refers to the velocity component and 'j' refers to the spatial
    component. Thus v_21 is

    :math:`$\frac{\partial w}{\partial y}$`

    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

        self.id = 'vgrad2D'
        self.tag = "vgrad2D"

        # setup the default properties if they do not exist
        dest_properties = dest.properties.keys()
        tensor_props = ["v_00", "v_01",
                        "v_10", "v_11"]

        for prop in tensor_props:
            if prop not in dest_properties:
                dest.add_property( dict(name=prop) )

        # set the to_reset variable to tensor_props as we want each of
        # the varaibles to be set to zero at the calc level
        self.to_reset = tensor_props

    def set_src_dst_reads(self):
        pass

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        """ Overide the basic function to call the tensor eval function """
        self.tensor_eval(kernel)

    cpdef tensor_eval(self, KernelBase kernel):

        # get the tag array pointer
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        # perform any iteration specific setup
        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        cdef size_t a

        cdef double result[4]

        cdef DoubleArray v_00 = self.dest.get_carray("v_00")
        cdef DoubleArray v_01 = self.dest.get_carray("v_01")

        cdef DoubleArray v_10 = self.dest.get_carray("v_10")
        cdef DoubleArray v_11 = self.dest.get_carray("v_11")

        for a in range( np ):
            if tag_arr.data[a] == LocalReal:
                self.eval_single(a, kernel, result)

                v_00.data[a] += result[0]
                v_01.data[a] += result[1]
                
                v_10.data[a] += result[2]
                v_11.data[a] += result[3]

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        
        cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        cdef size_t nnbrs = nbrs.length
        cdef size_t j

        # the particle itself does not contribute to the gradient
        if self.source is self.dest:
            nnbrs -= 1

        result[0] = result[1] = result[2] = result[3] = 0.0

        for j in range(nnbrs):
            self.eval_nbr( nbrs.data[j], dest_pid, kernel, result )

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double * result):

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef cPoint vba, grad, grada, gradb
        cdef double tmp

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        vba.x = self.s_u.data[source_pid] - self.d_u.data[dest_pid]
        vba.y = self.s_v.data[source_pid] - self.d_v.data[dest_pid]
        vba.z = self.s_w.data[source_pid] - self.d_w.data[dest_pid]

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = 0.5 * ( grada.x + gradb.x )
            grad.y = 0.5 * ( grada.y + gradb.y )
            grad.z = 0.5 * ( grada.z + gradb.z )

        else:
            grad = kernel.gradient( self._dst, self._src, 0.5*(ha+hb) )

        tmp = mb/rhob

        # the first row is for u,x and u,y
        result[0] += tmp * vba.x * grad.x
        result[1] += tmp * vba.x * grad.y

        # the second row is for v,x and v,y
        result[2] += tmp * vba.y * grad.x
        result[3] += tmp * vba.y * grad.y

################################################################################
# `KernelGradientCorrectionTerms` class.
################################################################################
cdef class BonnetAndLokKernelGradientCorrectionTerms(CSPHFunctionParticle):
    """ Evaluate the matrix terms eq(45) in "Variational and
    momentum preservation aspects of Smooth Particle Hydrodynamic
    formulations", Computer Methods in Applied Mechanical Engineering,
    180, (1997), 97-115

    The matrix would need to be inverted to calculate the correction terms!

    """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'kgc'

    def set_src_dst_reads(self):
        pass        

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid, 
                            KernelBase kernel, double *nr, double *dnr):
        cdef cPoint grada, gradb, grad
        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double Vb = mb/rhob

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)    
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint rba = cPoint_sub(self._src, self._dst)

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.x = (grada.x + gradb.x) * 0.5
            grad.y = (grada.y + gradb.y) * 0.5
            grad.z = (grada.z + gradb.z) * 0.5

            # grad.set((grada.x + gradb.x)*0.5,
            #          (grada.y + gradb.y)*0.5,
            #          (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        #m11
        nr[0] += Vb * grad.x * rba.x

        #m12 = m21
        nr[1] += Vb * grad.x * rba.y

        #m13 = m31
        nr[2] += Vb * grad.x * rba.z

        #m22
        dnr[0] += Vb * grad.y * rba.y
        
        #m23 = m32
        dnr[1] += Vb * grad.y * rba.z
        
        #m33
        dnr[2] += Vb * grad.z * rba.z

##########################################################################


################################################################################
# `FirstOrderCorrectionMatrix` class.
################################################################################
cdef class FirstOrderCorrectionMatrix(CSPHFunctionParticle):
    """ Kernel correction terms (Eq 14) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'liu-correction'

    def set_src_dst_reads(self):
        pass

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid, 
                            KernelBase kernel, double *nr, double *dnr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        tmp *= w

        nr[0] += tmp * rab.x * rab.x 
        nr[1] += tmp * rab.x * rab.y 
        nr[2] += tmp * rab.y * rab.y 

        dnr[0] -= tmp * rab.x
        dnr[1] -= tmp * rab.y
        dnr[2] -= tmp * rab.z

##########################################################################

################################################################################
# `FirstOrderKernelCorrectionTermsForAlpha` class.
################################################################################
cdef class FirstOrderCorrectionTermAlpha(SPHFunctionParticle):
    """ Kernel correction terms (Eq 15) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'alpha-correction'

    def set_src_dst_reads(self):
        pass

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.rkpm_d_beta1 = self.dest.get_carray("rkpm_beta1")
        self.rkpm_d_beta2 = self.dest.get_carray("rkpm_beta2")
        self.rkpm_d_alpha = self.dest.get_carray("rkpm_alpha")
      
        self.rkpm_d_dbeta1dx = self.dest.get_carray("rkpm_dbeta1dx")
        self.rkpm_d_dbeta1dy = self.dest.get_carray("rkpm_dbeta1dy")

        self.rkpm_d_dbeta2dx = self.dest.get_carray("rkpm_dbeta2dx")
        self.rkpm_d_dbeta2dy = self.dest.get_carray("rkpm_dbeta2dy")

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w, beta, tmp1, tmp2, tmp3, Vb
        
        cdef cPoint rab, grad

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        beta1 = self.rkpm_d_beta1.data[dest_pid]
        beta2 = self.rkpm_d_beta2.data[dest_pid]

        dbeta1dx = self.rkpm_d_dbeta1dx[dest_pid]
        dbeta1dy = self.rkpm_d_dbeta1dy[dest_pid]

        dbeta2dx = self.rkpm_d_dbeta2dx[dest_pid]
        dbeta2dy = self.rkpm_d_dbeta2dy[dest_pid]

        alpha = self.rkpm_d_alpha[dest_pid]

        rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        Vb = mb/rhob
        
        grad = kernel.gradient(self._dst, self._src, h)

        tmp3 = Vb*(1.0 + (beta1*rab.x + beta2*rab.y))
        
        tmp1 = Vb*w* (dbeta1dx*rab.x + beta1 + rab.y*dbeta2dx) + tmp3*grad.x

        tmp2 = Vb*w* (dbeta1dy*rab.x + beta2 + rab.y*dbeta2dy) + tmp3*grad.y
        
        #alpha
        nr[0] += Vb*w * (1 + (beta1*rab.x + beta2*rab.y))

        #dalphadx
        nr[1] += -tmp1

        #dalphady
        nr[2] += -tmp2

################################################################################
# `FirstOrderCorrectionMatrixGradient` class.
################################################################################
cdef class FirstOrderCorrectionMatrixGradient(CSPHFunctionParticle):
    """ Kernel correction terms (Eq 15) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """
    def set_src_dst_reads(self):
        pass

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w, beta, Vb
        
        cdef cPoint rab, grad

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        Vb = mb/rhob

        grad = kernel.gradient(self._dst, self._src, h)
        
        nr[0] += 2*Vb*w*rab.x + Vb*rab.x*rab.x*grad.x

        nr[1] += Vb*rab.x*rab.x*grad.y

        nr[2] += Vb*w*rab.y + Vb*rab.y*rab.x*grad.x

        dnr[0] += Vb*rab.x*(rab.y*grad.y + w)

        dnr[1] += Vb*rab.y*rab.y*grad.x

        dnr[2] += 2*Vb*rab.y*w + Vb*rab.y*rab.y*grad.y

##########################################################################

################################################################################
# `FirstOrderCorrectionVectorGradient` class.
################################################################################
cdef class FirstOrderCorrectionVectorGradient(CSPHFunctionParticle):
    """ Kernel correction terms (Eq 15) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """

    def set_src_dst_reads(self):
        pass

    #Defined in the .pxd file
    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w, Vb
        
        cdef cPoint rab, grad

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        Vb = mb/rhob        
        
        grad = kernel.gradient(self._dst, self._src, h)
        
        nr[0] += -Vb*rab.x*grad.x - Vb*w

        nr[1] += -Vb*rab.x*grad.y

        nr[2] += -Vb*rab.y*grad.x

        dnr[0] += -Vb*rab.y*grad.y - Vb*w

##########################################################################


##############################################################################
# `KernelSum`
##############################################################################
cdef class KernelSum(SPHFunctionParticle):
    def set_src_dst_reads(self):
        self.src_reads = ['h', 'x', 'y', 'z', 'rho', 'm']
        self.dst_reads = ['h', 'x', 'y', 'z']

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        cdef double w = kernel.function(self._dst, self._src, h)

        nr[0] += (w * self.s_m.data[source_pid] / self.s_rho.data[source_pid])
