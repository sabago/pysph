"""Cython functions to solve the time dependent Euler equations. """

import numpy
cimport numpy

cdef extern from "math.h":
    double fabs(double)
    double sqrt(double)

cdef inline double max(double a, double b):
    if a < b:
        return b
    else:
        return a

cdef inline double min(double a, double b):
    if a < b:
        return a
    else:
        return b

cpdef compute_star(int dest_pid, int source_pid,
                   double gamma,
                   double rhol, double rhor,
                   double pl, double pr,
                   double ul, double ur):    

    cdef double tmp1 = 1.0/(2*gamma)
    cdef double tmp2 = 1.0/(gamma - 1.0)
    cdef double tmp3 = 1.0/(gamma + 1.0)

    cdef double gamma1 = (gamma - 1.0) * tmp1
    cdef double gamma2 = (gamma + 1.0) * tmp1
    cdef double gamma3 = 2*gamma * tmp2
    cdef double gamma4 = 2 * tmp2
    cdef double gamma5 = 2 * tmp3
    cdef double gamma6 = tmp3/tmp2
    cdef double gamma7 = 0.5 * (gamma - 1.0)
    cdef double gamma8 = gamma - 1.0

    cdef double cl = sqrt(gamma*pl/rhol)
    cdef double cr = sqrt(gamma*pr/rhor)

    # constants for the Iteration
    cdef double tol = 1e-6
    cdef int niter = 20

    cdef double change
    cdef int i

    # check the initial data
    if ( gamma4*(cl+cr) <= (ur-ul) ):
        raise RuntimeError("Vaccum generated. Exiting")

    # compute the exact solution for the pressure and velocity in the star region
    cdef double pm, um
    
    cdef double qser = 2.0

    # compute the guess pressure 'pm'
    cdef double cup = 0.25 * (rhol + rhor)*(cl+cr)
    cdef double ppv = 0.5 * (pl + pr) + 0.5*(ul - ur)*cup
    cdef double pmin = min(pl, pr)
    cdef double pmax = max(pl, pr)
    cdef double qmax = pmax/pmin

    ppv = max(0.0, ppv)

    if ( (qmax <= qser) and (pmin <= ppv) and (ppv <= pmax) ):
        pm = ppv

    elif ( ppv < pmin ):
        
        pq = (pl/pr)**gamma1
        um = ( pq*ul/cl + ur/cr + gamma4*(pq - 1.0) )/( pq/cl + 1.0/cr )
        ptl = 1.0 + gamma7 * (ul - um)/cl
        ptr = 1.0 + gamma7 * (um - ur)/cr
        pm = 0.5*(pl*ptl**gamma3 + pr*ptr**gamma3)
    
    else:

        gel = sqrt( (gamma5/rhol)/(gamma6*pl + ppv) )
        ger = sqrt( (gamma5/rhor)/(gamma6*pr + ppv) )
        pm = (gel*pl + ger*pr - (ur-ul))/(gel + ger)

    # the guessed value is pm
    pstart = pm

    pold = pstart
    udifff = ur-ul

    for i in range(niter):
        fl, fld = prefun(pold, rhol, pl, cl, gamma)
        fr, frd = prefun(pold, rhor, pr, cr, gamma)

        p = pold - (fl + fr + udifff)/(fld + frd)
        change = 2.0 * fabs((p-pold)/(p+pold))
        if change <= tol:
            break
        pold = p

    if i == niter - 1:
        print dest_pid, source_pid, rhol, rhor, pl, pr, ul, ur
        raise RuntimeError("Divergence in Newton-Raphson Iteration")

    # compute the velocity in the star region 'um'
    um = 0.5 * (ul + ur + fr - fl)
    return p, um
        
cdef prefun(double p, double dk, double pk, double ck, double gamma):

    cdef double g1 = (gamma - 1.0)/(2*gamma)
    cdef double g2 = (gamma + 1.0)/(2.0*gamma)
    cdef double g4 = 2.0/(gamma - 1.0)
    cdef double g5 = 2.0/(gamma + 1.0)
    cdef double g6 = (gamma - 1.0)/(gamma + 1.0)

    cdef double f, fd
    cdef double pratio, ak, bk, qrt
    
    if (p <= pk):
        pratio = p/pk
        f = g4*ck*(pratio**g1 - 1.0)
        fd = (1.0/(dk*ck))*pratio**(-g2)

    else:
        ak = g5/dk
        bk = g6*pk
        qrt = sqrt(ak/(bk+p))
        f = (p-pk)*qrt
        fd = (1.0 - 0.5*(p-pk)/(bk + p))*qrt

    return f, fd

cpdef sample(double pm, double um, double s,
             double rhol, double rhor,
             double pl, double pr,
             double ul, double ur,
             double gamma):
    
    cdef double rho, u, p
    
    cdef double cl = sqrt(gamma*pl/rhol)
    cdef double cr = sqrt(gamma*pr/rhor)

    cdef double tmp1 = 1.0/(2*gamma)
    cdef double tmp2 = 1.0/(gamma - 1.0)
    cdef double tmp3 = 1.0/(gamma + 1.0)

    cdef double g1 = (gamma - 1.0) * tmp1
    cdef double g2 = (gamma + 1.0) * tmp1
    cdef double g3 = 2*gamma * tmp2
    cdef double g4 = 2 * tmp2
    cdef double g5 = 2 * tmp3
    cdef double g6 = tmp3/tmp2
    cdef double g7 = 0.5 * (gamma - 1.0)
    cdef double g8 = gamma - 1.0

    if s <= um:
        # sampling point lies to the left of the contact discontinuity
        if (pm <= pl):
            # left rarefaction
            shl = ul - cl

            if (s <= shl):
                # sampled point is left state
                rho = rhol; u = ul; p = pl
            else:
                cml = cl*(pm/pl)**g1
                stl = um - cml

                if (s > stl):
                    # sampled point is Star left state
                    rho = rhol*(pm/pl)**(1.0/gamma); u = um; p = pm
                else:
                    
                    # sampled point is inside left fan
                    u = g5 * (cl + g7*ul + s)
                    c = g5 * (cl + g7*(ul - s))

                    rho = rhol*(c/cl)**g4
                    p = pl * (c/cl)**g3

        else: # pm <= pl
            # left shock
            pml = pm/pl
            sl = ul - cl*sqrt(g2*pml + g1)

            if (s <= sl):
                # sampled point is left data state
                rho = rhol; u=ul; p=pl
            else:
                # sampled point is Star left state
                rho = rhol*(pml + g6)/(pml*g6 + 1.0)
                u = um
                p = pm

    else: # s < um

        # sampling point lise to the right of the contact discontinuity
        if (pm > pr):

            # right shock
            pmr = pm/pr
            sr = ur + cr * sqrt(g2*pmr + g1)

            if (s >= sr):
                #sampled point is right data state
                rho = rhor; u = ur; p = pr
            else:
                # sampled point is star right state
                rho = rhor*(pmr + g6)/(pmr*g6 + 1.0); u = um; p = pm
        else:
            # right rarefaction
            shr = ur + cr

            if (s >= shr):
                # sampled point is right state
                rho = rhor; u = ur; p = pr
            else:
                cmr = cr*(pm/pr)**g1
                STR = um + cmr

                if (s <= STR):
                    # sampled point is star right
                    rho = rhor*(pm/pr)**(1.0/gamma); u = um; p = pm
                else:
                    # sampled point is inside left fan
                    u = g5*(-cr + g7*ur + s)
                    c = g5*(cr - g7*(ur -s))
                    rho = rhor * (c/cr)**g4
                    p = pr*(c/cr)**g3
                    
    return rho, u, p
