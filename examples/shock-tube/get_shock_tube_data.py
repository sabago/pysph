""" Functions to get the initial data for the shock tube problems """

import numpy

import pysph.base.api as base

def get_shock_tube_data(nl, nr, xl, xr,
                        pl, pr, rhol, rhor, ul, ur,
                        g1, g2, h0, gamma=1.4,
                        m0=None):

    dxl = numpy.abs(xl)/nl
    dxr = numpy.abs(xr)/nr

    x = numpy.ones( nl + nr )
    x[:nl] = numpy.arange( xl, -dxl+1e-10, dxl )
    x[nl:] = numpy.arange( dxr, +xr+1e-10, dxr )

    p = numpy.ones_like(x)
    p[:nl] = pl
    p[nl:] = pr

    rho = numpy.ones_like(x)
    rho[:nl] = rhol
    rho[nl:] = rhor

    u = numpy.ones_like(x)
    u[:nl] = ul
    u[nl:] = ur

    e = p/( (gamma-1)*rho )
    cs = numpy.sqrt( gamma*p/rho )

    if not m0:
        m = numpy.ones_like(x) * dxl
    else:
        m = numpy.ones_like(x) * m0
        
    h = numpy.ones_like(x) * h0

    # Extra properties for the ADKE procedure
    rhop = numpy.ones_like(x)
    div = numpy.ones_like(x)
    q = g1 * h * cs

    adke = base.get_particle_array(name="fluid", x=x, m=m, rho=rho, h=h,
                              u=u, p=p, e=e, cs=cs,
                              rhop=rhop, div=div, q=q)

    nbp = 100
        
    # left boundary
    x = numpy.ones(nbp)
    for i in range(nbp):
        x[i] = xl - (i + 1) * dxl

    if not m0:
        m = numpy.ones_like(x) * dxl
    else:
        m = numpy.ones_like(x) * m0
        
    h = numpy.ones_like(x) * h0

    u = numpy.zeros_like(x) * ul
    rho = numpy.ones_like(x) * rhol
    p = numpy.ones_like(x) * pl

    e = p/( (gamma-1) * rho )
    cs = numpy.sqrt( gamma * p/rho )
    q = h * cs * g1

    left = base.get_particle_array(name="left", x=x, m=m, h=h, u=u,
                                   type=base.Boundary,
                                   rho=rho, p=p, e=e, cs=cs, q=q)
    
    # right boundary
    x = numpy.ones(nbp)
    for i in range(nbp):
        x[i] = xr + (i + 1) * dxr

    if not m0:
        m = numpy.ones_like(x) * dxl
    else:
        m = numpy.ones_like(x) * m0
        
    h = numpy.ones_like(x) * h0

    u = numpy.zeros_like(x) * ur
    rho = numpy.ones_like(x) * rhor
    p = numpy.ones_like(x) * pr

    e = p/( (gamma-1)*rho )

    cs = numpy.sqrt( gamma * p/rho )
    q = h * cs * g1

    right = base.get_particle_array(name="right", x=x, m=m, h=h, u=u,
                                    type=base.Boundary,
                                    rho=rho, p=p, e=e, cs=cs, q=q)

    return adke, left, right
