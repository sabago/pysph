cdef extern from "math.h":
    double sqrt(double)
    double fabs(double)

cdef inline double max(double a, double b)

cdef inline double compute_signal_velocity(double beta, double vabdotj,
                                           double ca, double cb)

cdef inline double compute_signal_velocity2(double beta, double vabdotj,
                                            double ca, double cb)
