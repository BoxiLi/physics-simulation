import numpy as np
import vtktools 
from functools import reduce
from itertools import product as iter_product
"""
traj[time_slice, particle_index, dim]
"""
class particle_dynamics_sim(object):
    def __init__(self, dt, iter_num, par_num, dim, box=(5.,5.,5.), inter_range = None):
        self. dt = dt
        self.iter_num = iter_num
        self.par_num = par_num
        self.dim = dim
        self.traj = np.zeros([iter_num+2, par_num, dim])
        self.v0 = np.zeros([par_num, dim])
        self.initialized = False
        self.box = np.array(box)
        self.linked_cell = None
        self.inter_range = inter_range

    def initialize(self, start_pos, start_v):
        self.traj[0] = start_pos
        self.v0 = start_v
        self.initialized = True


    def run(self):
        """
        Use Verlet integral method to do the simulation, return the trajectory in the form of
        traj[time_slice, particle_index, dim]
        """
        global traj0
        global traj1
        global traj2 
        global glo_force
        if self.inter_range is not None:
            cell_shape = (self.box//self.inter_range).astype(int)
            for k in range(len(cell_shape)):
                if cell_shape[k]==0:
                    cell_shape[k] = 1
            cell_size = self.box/cell_shape
        else:
            cell_shape = None
            cell_size = None

        vtk_writer = vtktools.VTK_XML_Serial_Unstructured()
        
        if not self.initialized:
            # create random points inthe box
            self.traj[0] = np.random.rand(self.par_num, self.dim) * self.box
            self.v0 = np.zeros((self.par_num, self.dim), dtype=float)

        self.traj[1] = np.mod(self.traj[0] + self.v0, self.box)
        next = 2
        while next < self.traj.shape[0]:
            # calculate force
            if self.inter_range is not None:
                self.linked_cell = create_linked_cell(self.traj[next-1], self.box, self.inter_range)
            force = LJ_force(self.traj[next-1], self.linked_cell, cell_shape, cell_size)
            # print(force)
            self.traj[next] = 2*self.traj[next-1] - self.traj[next-2] + self.dt**2*force 
            # periodic boundary        
            self.traj[next] = np.mod(self.traj[next], self.box) #??????????????????? mod could be a problem

            #vktools
            r_x = self.traj[next][:,0]
            r_y = self.traj[next][:,1]
            r_z = self.traj[next][:,2]
            F_x = force[:,0]
            F_y = force[:,1]
            F_z = force[:,2]
            vtk_writer.snapshot("simu/MD_"+ str(self.dt) + ".vtu", r_x,r_y,r_z,
                x_force = F_x, y_force = F_y, z_force = F_z)
            
            if 0.0 in self.traj[next]:
                print(self.traj[next-1])
                print(self.traj[next])
                print(force)
                traj0 = self.traj[next-2]
                traj1 = self.traj[next-1]
                traj2 = self.traj[next]
                glo_force = force
                assert(False)

            next += 1

        # vtk_writer.writePVD("MD.pvd")
        return self.traj


def get_cell_ind(cell_coor, cell_shape):
    """
    Convert n-D cell coordinate to 1D cell index
    """
    cell_ind = 0
    dim = len(cell_coor)
    for i in range(dim-1):
        cell_ind += cell_coor[i] * reduce(lambda x,y:x*y, cell_shape[i+1:])
    cell_ind += cell_coor[-1]
    return cell_ind


def create_linked_cell(pos_all, box, r):
    box = np.array(box)
    # how many cells in each dimension
    cell_shape = (box//r).astype(int)
    for k in range(len(cell_shape)):
        if cell_shape[k]==0:
            cell_shape[k] = 1
    # the size of one cell
    cell_size = box/cell_shape
    # total number of cells
    head_length = reduce(lambda x,y:x*y, cell_shape)

    # allocate particles to different cells
    cell_allocation = [[] for i in range(head_length)]
    for par_ind in range(pos_all.shape[0]):
        pos = pos_all[par_ind]
        # n-D coordinate of the cell
        cell_coor = (pos//cell_size).astype(int)
        # scalar coordinate of the cell
        cell_ind = get_cell_ind(cell_coor, cell_shape)
        cell_allocation[cell_ind].append(par_ind)

    # create head and first
    head = np.empty(head_length, dtype = int)
    first = np.empty(pos_all.shape[0], dtype = int)
    # max_int is used to mark the end of a cell
    max_int = np.iinfo(int).max
    for i in range(head_length):
        par_list = cell_allocation[i]
        if not par_list: # if no particle in this cell
            head[i] = max_int
        else:
            current = par_list.pop()
            head[i] = current
            while par_list:
                first[current] = par_list.pop()
                current = first[current]
            first[current] = max_int

    return [head, first]


def find_cell_par(linked_cell, cell_ind):
    current = linked_cell[0][cell_ind]
    result = []
    while current!=np.iinfo(int).max:
        result.append(current)
        current = linked_cell[1][current]
    return result


def find_near_par(linked_cell, cell_coor, cell_shape):
    # find coordinate of all neighbours cell for the use of itertools product
    # [[a-1,a,a+1],[b-1,b,b+1],...]
    dim = len(cell_coor)
    neighbour_coor = np.empty((dim, 3), dtype = int)
    for axis in range(dim):
        coor = cell_coor[axis]
        neighbour_coor[axis] = np.mod([coor-1,coor,coor+1], cell_shape[axis])

    # convert to 1D cell index and remove repeatition
    cell_ind_set = set()
    for cell_coor in iter_product(*neighbour_coor):
        cell_ind_set.add(get_cell_ind(cell_coor, cell_shape))

    # find the particles in the neighbours
    search_range = []
    for cell_ind in cell_ind_set:
        search_range += find_cell_par(linked_cell, cell_ind)
    return search_range



def LJ(r, direction):
    if r==0.0:
        # print("identical partical!")
        return 0
    else:
        return 4 * (-12*r**(-13) + 6*r**(-7)) * direction


def LJ_force(pos_all, linked_cell, cell_shape, cell_size):
    """
    Lennard-Jones potential
    """
    par_num = pos_all.shape[0]
    force = np.zeros(pos_all.shape, dtype=np.float64)
    for par_index1 in range(par_num):
        pos1 = pos_all[par_index1]
        if linked_cell is None:
            search_range = list(range(par_num))
        else:
            cell_coor = (pos1//cell_size).astype(int)
            search_range = find_near_par(linked_cell, cell_coor, cell_shape)
            # search_range.sort() # why !!!!!!!!!!!?
        for par_index2 in search_range:
            if par_index1 != par_index2:
                pos2 = pos_all[par_index2]
                dist = pos1-pos2
                r = np.linalg.norm(dist)
                if r==0.0:
                    print("identical partical!")
                    continue
                force[par_index1] += -LJ(r, dist/r)
    tesmp_force = force

    force = np.zeros(pos_all.shape, dtype=np.float64)
    for par_index1 in range(par_num):
        pos1 = pos_all[par_index1]
        if linked_cell is None:
            search_range = list(range(par_num))
        else:
            cell_coor = (pos1//cell_size).astype(int)
            search_range = find_near_par(linked_cell, cell_coor, cell_shape)
            search_range.sort() # why !!!!!!!!!!!?
        for par_index2 in search_range:
            if par_index1 != par_index2:
                pos2 = pos_all[par_index2]
                dist = pos1-pos2
                r = np.linalg.norm(dist)
                if r==0.0:
                    print("identical partical!")
                    continue
                force[par_index1] += -LJ(r, dist/r)
                
    diff = np.max(tesmp_force-force)
    if diff>1.0e-10:
        print(diff)
    return force


def test_small():
    # test get_cell_ind
    cell_shape = (2,3,4)
    cell_coor = (1,1,1)
    assert(get_cell_ind(cell_coor, cell_shape)==17)

    # test create_linked_cell
    box = np.array((2.,2.))
    r = 1.
    cell_shape = (box//r).astype(int)
    pos_all = np.array([
            [1.2,0.5],
            [0.2,1.3],
            [1.1,0.8],
            [0.9,1.2],
            [1.8,0.7],
            [1.5,1.2],
            [1.8,1.6],
            [0.6,1.8]
    ])
    linked_cell = create_linked_cell(pos_all, box, r)
    max_int = np.iinfo(int).max
    head = np.array([max_int,7,4,6])
    first = np.array([max_int, max_int,0,1,2,max_int,5,3])
    assert(np.array_equal(linked_cell[0],head))
    assert(np.array_equal(linked_cell[1],first))

    # test find_cell_par
    result = find_cell_par(linked_cell, 2)
    result.sort()
    assert((result == [0,2,4]))

    # test find_near_par
    result = find_near_par(linked_cell, (0,1), cell_shape)
    result.sort()
    assert(result == [0,1,2,3,4,5,6,7])
# test_small()

global traj0
global traj1
global traj2 
global glo_force
    

# np.random.seed(0)
# test = particle_dynamics_sim(dt=0.001, iter_num=1000, par_num = 10, dim = 3, box = (3.,3.,3.))
# result = test.run()
# print(result[-1])

np.random.seed(0) 
test = particle_dynamics_sim(dt=0.001, iter_num=1000, par_num = 10, dim = 3, box = (3.,3.,3.), inter_range=np.inf)
result = test.run()
print(result[-1])

# start_traj = np.array(
# [[1.54264129, 0.0415039,  1.26729647],
# [1.49760777, 0.99701402, 0.44959329],
# [0.39612573, 1.52106142, 0.33822167],
# [0.17667963, 1.37071964, 1.90678669],
# [0.00789653, 1.02438453, 1.62524192],
# [1.22505213, 1.44351063, 0.58375214],
# [1.83554825, 1.42915157, 1.08508874],
# [0.2843401,  0.74668152, 1.34826723],
# [0.88366635, 0.86802799, 1.23553396],
# [1.02627649, 1.30079436, 1.20207791]])