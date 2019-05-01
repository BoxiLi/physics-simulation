import numpy as np
import vtktools 
"""
traj[time_slice, particle_index, dim]
"""
class particle_dynamics_sim(object):
    def __init__(self, dt, iter_num, par_num, dim, box=(5.,5.,5.)):
        self. dt = dt
        self.iter_num = iter_num
        self.par_num = par_num
        self.dim = dim
        self.traj = np.empty([iter_num+2, par_num, dim])
        self.v0 = np.empty([par_num, dim])
        self.initialized = False
        self.box = box
        self.linked_cell = None

    def initialize(self, start_pos, start_v):
        self.traj[0] = start_pos
        self.v0 = start_v
        self.initialized = True


    def run(self):
        """
        Use Verlet integral method to do the simulation, return the trajectory in the form of
        traj[time_slice, particle_index, dim]
        """
        vtk_writer = vtktools.VTK_XML_Serial_Unstructured()
        
        if not self.initialized:
            # create random points inthe box
            self.traj[0] = np.random.rand(self.par_num, self.dim) * self.box
            self.v0 = np.zeros((self.par_num, self.dim), dtype=float)

        self.traj[1] = np.mod(self.traj[0] + self.v0, self.box)
        next = 2
        while next < self.traj.shape[0]:
            force = LJ_force(self.traj[next-1])
            # print(force)
            self.traj[next] = 2*self.traj[next-1] - self.traj[next-2] + self.dt**2*force
            # periodic boundary        
            self.traj[next] = np.mod(self.traj[next], self.box)
            # print(force)
            # print(self.traj[next-1])
            #vktools
            r_x = self.traj[next][:,0]
            r_y = self.traj[next][:,1]
            r_z = self.traj[next][:,2]
            F_x = force[:,0]
            F_y = force[:,1]
            F_z = force[:,2]
            vtk_writer.snapshot("simu/MD_"+ str(self.dt) + ".vtu", r_x,r_y,r_z,
                x_force = F_x, y_force = F_y, z_force = F_z)

            # print(np.max(self.traj[next]-self.traj[next-1]))
            next += 1

        # vtk_writer.writePVD("MD.pvd")
        # print(self.traj[-1])
        return self.traj

def find_cell():
    pass


def find_near(linked_cell):
    pass


def LJ(r, direction):
    if r==0.0:
        print("identical partical!")
        return 0
    else:
        return 4 * (-12*r**(-13) + 6*r**(-7)) * direction


def LJ_force(pos_all, linked_cell=None):
    """
    Lennard-Jones potential
    """
    par_num = pos_all.shape[0]
    force = np.zeros(pos_all.shape)
    for par_index1 in range(par_num):
        if linked_cell is None:
            search_range = list(range(par_num))
        else:
            search_range = find_near(linked_cell)
        for par_index2 in search_range:
            if par_index1 != par_index2:
                pos1 = pos_all[par_index1]
                pos2 = pos_all[par_index2]
                dist = pos1-pos2
                r = np.linalg.norm(dist)
                force[par_index1] = -LJ(r, dist/r)
                # print(force)

    return force
    

start_traj =np.array(
  [[1.54264129, 0.0415039,  1.26729647],
  [1.49760777, 0.99701402, 0.44959329],
  [0.39612573, 1.52106142, 0.33822167],
  [0.17667963, 1.37071964, 1.90678669],
  [0.00789653, 1.02438453, 1.62524192],
  [1.22505213, 1.44351063, 0.58375214],
  [1.83554825, 1.42915157, 1.08508874],
  [0.2843401,  0.74668152, 1.34826723],
  [0.88366635, 0.86802799, 1.23553396],
  [1.02627649, 1.30079436, 1.20207791]])

np.random.seed(0)   
test = particle_dynamics_sim(dt=0.001, iter_num=10, par_num = 10, dim = 3, box = (2.,2.,2.))
# test.initialize(start_traj, np.zeros((10,3)))
result = test.run()