""" Tests for the parallel cell manager """

import nose.plugins.skip as skip
raise skip.SkipTest("Dont run this test via nose")

from pysph.parallel.simple_block_manager import SimpleBlockManager
from pysph.base.particles import Particles
from pysph.base.particle_array import get_particle_array
from pysph.base.point import IntPoint

import numpy
import pylab
import time

# mpi imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = pid = comm.Get_rank()

def draw_cell(cell, color="b"):
    centroid = base.Point()
    cell.get_centroid(centroid)
    
    half_size = 0.5 * cell.cell_size

    x1, y1 = centroid.x - half_size, centroid.y - half_size
    x2, y2 = x1 + cell.cell_size, y1
    x3, y3 = x2, y1 + cell.cell_size
    x4, y4 = x1, y3

    pylab.plot([x1,x2,x3,x4,x1], [y1, y2, y3, y4,y1], color)

def draw_block(origin, block_size, block_id, color="r"):

    half_size = 0.5 * block_size
    x,y = [], []

    xc = origin.x + ((block_id.x + 0.5) * proc_map.block_size)        
    yc = origin.y + ((block_id.y + 0.5) * proc_map.block_size)
        
    x1, y1 = xc - half_size, yc - half_size
    x2, y2 = x1 + block_size, y1
    x3, y3 = x2, y2 + block_size
    x4, y4 = x1, y3
    
    pylab.plot([x1,x2,x3,x4,x1], [y1, y2, y3, y4,y1], color)

def draw_particles(cell, color="y"):
    arrays = cell.arrays_to_bin
    num_arrays = len(arrays)
    
    index_lists = []
    cell.get_particle_ids(index_lists)

    x, y = [], []

    for i in range(num_arrays):
        array = arrays[i]
        index_array = index_lists[i]
        
        indices = index_lists[i].get_npy_array()

        xarray, yarray = array.get('x','y')
        for j in indices:
            x.append(xarray[j])
            y.append(yarray[j])

    pylab.plot(x,y,color+"o")

def get_sorted_indices(cell):
    index_lists = []
    cell.get_particle_ids(index_lists)
    index_array = index_lists[0].get_npy_array()
    index_array.sort()

    print type(index_array)
    return index_array

if pid == 0:
    x = numpy.array( [0, 0.2, 0.4, 0.6, 0.8] * 5 )
    y = numpy.array( [0.0, 0.0, 0.0, 0.0, 0.0,
                      0.2 ,0.2, 0.2, 0.2, 0.2,
                      0.4, 0.4, 0.4, 0.4, 0.4,
                      0.6, 0.6, 0.6, 0.6, 0.6,
                      0.8, 0.8, 0.8, 0.8, 0.8] )
    x += 1e-10
    y += 1e-10
    
    h = numpy.ones_like(x) * 0.3/2.0

    block_00 = 0, 1, 5, 6
    block_10 = 2, 7
    block_20 = 3, 4, 8, 9
    block_01 = 10, 11
    block_11 = 12
    block_21 = 13, 14
    block_02 = 15, 16, 20, 21
    block_12 = 17, 22
    block_22 = 18, 19, 23, 24

    cids = [block_00, block_10, block_20,
            block_01, block_11, block_21,
            block_02, block_12, block_22]

if pid == 1:
    x = numpy.array( [0.8, 1.0, 1.2, 1.4, 1.6] * 5 )
    y = numpy.array( [0.0, 0.0, 0.0, 0.0, 0.0,
                      0.2, 0.2, 0.2, 0.2, 0.2,
                      0.4, 0.4, 0.4, 0.4, 0.4,
                      0.6, 0.6, 0.6, 0.6, 0.6,
                      0.8, 0.8, 0.8, 0.8, 0.8] )

    x += 1e-10
    y += 1e-10

    h = numpy.ones_like(x) * 0.3/2.0

    block_20 = 4, 9
    block_30 = 1, 6
    block_40 = 2, 3, 7, 8
    block_50 = 4, 9
    block_21 = 14
    block_31 = 11
    block_41 = 12, 13
    block_51 = 14
    block_22 = 15, 20
    block_32 = 16, 21
    block_42 = 17, 18, 22, 23
    block_52 = 19, 24

    cids = [block_20, block_30, block_40, block_50,
            block_21, block_31, block_41, block_51,
            block_22, block_32, block_42, block_52]
    

pa = get_particle_array(name="test"+str(rank), x=x, y=y, h=h)
particles = Particles(arrays=[pa,])

# create the block manager
pm = pm = SimpleBlockManager(block_scale_factor=2.0)
pm.initialize(particles)

cm = pm.cm

assert ( abs(pm.block_size - 0.3) < 1e-15 )
assert (pm.block_size == cm.cell_size)


cells_dict = cm.cells_dict
pmap = pm.processor_map

assert (len(cells_dict) == len(cids))
    
# call an update
pm.update()

# test the processor map's local and global cell map
local_cell_map = pmap.local_cell_map
global_cell_map = pmap.global_cell_map

assert (len(local_cell_map) == len(cells_dict))
for cid in local_cell_map:
    assert( cid in cells_dict )
    assert( list(local_cell_map[cid])[0] == rank )
    
if rank == 0:
    other_cids = comm.recv(source=1)
    comm.send(cids, dest=1)

if rank == 1:
    comm.send(cids, dest=0)
    other_cids = comm.recv(source=0)

conflicting_cells = IntPoint(2,0,0), IntPoint(2,1,0), IntPoint(2,2,0)

# check the conflicting cells
for cid in conflicting_cells:
    assert ( cid in global_cell_map )
    pids = list(global_cell_map[cid])
    pids.sort()

    assert ( pids == [0,1] )

# check the cells_to_send_list
cells_to_send = pmap.get_cell_list_to_send()
if rank == 0:
    expected_list = [IntPoint(1,0), IntPoint(1,1), IntPoint(1,2),
                     IntPoint(2,0), IntPoint(2,1), IntPoint(2,2)]
    
    cell_list = cells_to_send[1]
    
if rank == 1:
    expected_list = [IntPoint(2,0), IntPoint(2,1), IntPoint(2,2),
                     IntPoint(3,0), IntPoint(3,1), IntPoint(3,2)]

    cell_list = cells_to_send[0]

for cid in expected_list:
    assert (cid in cell_list)
    
pa = pm.arrays[0]
print rank, pa.num_real_particles, pa.get_number_of_particles()
