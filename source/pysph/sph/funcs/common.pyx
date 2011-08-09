cdef inline double max(double a, double b):
    if a < b:
        return b
    else:
        return a

cdef inline double compute_signal_velocity(double beta, double vabdotj,
                                           double ca, double cb):
    
    cdef double tmp1 = sqrt(ca*ca + beta*vabdotj*vabdotj)
    cdef double tmp2 = sqrt(cb*cb + beta*vabdotj*vabdotj)
    return tmp1 + tmp2 - vabdotj

cdef inline double compute_signal_velocity2(double beta, double vabdotj,
                                            double ca, double cb):
    
    return (ca + cb - 2*vabdotj)

