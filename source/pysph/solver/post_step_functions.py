""" Post step functions for the solver """

import pickle
import os

import pysph.base.api as base
from pysph.base.cell import py_find_cell_id

class SaveCellManagerData(object):
    """Post-step function to save the cell manager's data.

    Two files are created, 'neighbors' contains partile neighbor
    information as returned by the neighbor locator. For each
    particle, a LongArray for it's neighbor indices are stored.

    The second file
    'cells', holds cell data for each cell (partilce indices, coordinates)

    """

    def __init__(self, rank = 0, path=None, count=10):
        self.rank = rank

        self.count = count
        
        if path:
            self.path = path
        else:
            self.path = "."
    
    def eval(self, solver):
        
        if not ((solver.count % self.count) == 0):
            return

        particles = solver.particles
        time = solver.t

        nnps = particles.nnps_manager
        locator_cache = nnps.particle_locator_cache
        
        num_locs = len(locator_cache)
        locators = locator_cache.values()

        fname_base = os.path.join(self.path+"/neighbors_"+str(self.rank))

        cell_manager = particles.cell_manager
        cell_size = cell_manager.cell_size

        neighbor_idx = {}

        for i in range(num_locs):
            loc = locators[i]
            dest = loc.dest
            src = loc.source
            
            particle_indices = dest.get('idx')

            x, y, z = dest.get("x", "y", "z")

            neighbor_idx[dest.name + '-' + src.name] = {}
            d = neighbor_idx[dest.name + '-' + src.name]

            nrp = dest.num_real_particles

            for j in range(nrp):
                neighbors = loc.py_get_nearest_particles(j)
                
                temp = dest.extract_particles(neighbors)
                particle_idx = particle_indices[j]
                
                pnt = base.Point(x[j], y[j], z[j])
                cid = py_find_cell_id(pnt, cell_size)

                idx = temp.get_carray("idx")

                d[particle_idx] = {'neighbors':idx, 'cid':cid}
            
            fname = fname_base + "_" + dest.name + "_" + str(solver.count)

            # save particle neighbor information.
            f = open(fname, 'w')
            pickle.dump(neighbor_idx, f)
            f.close()
            
            fname_cells = os.path.join(self.path+"/cells_"+str(self.rank))
            fname_cells += "_" + str(solver.count)

            # ask the cell manager to save the particle representation
            cell_manager.get_particle_representation(fname_cells)
            

class CFLTimeStepFunction(object):
    def __init__(self, CFL=0.3):
        self.cfl = CFL

    def eval(self, solver):
        v = float('inf')
        for pa in solver.particles.arrays:
            val = min(pa.h/(pa.cs+(pa.u**2+pa.v**2+pa.w**2)**0.5))
            if val < v:
                v = val
        solver.dt = self.cfl*v
