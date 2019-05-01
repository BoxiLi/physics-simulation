import numpy as np
import vtktools 
"""
traj[time_slice, particle_index, dim]
"""
class particle_dynamics_sim(object):
    def __init__(self, dt, iter_num, par_num, dim, box=(5.,5.,5.)):
        self.dt = dt
        self.iter_num = iter_num
        self.par_num = par_num
        self.dim = dim
        self.traj = np.empty([iter_num+2, par_num, dim])
        self.v0 = np.empty([par_num, dim])
        self.initialized = False
        self.box = box


    def run(self):
        """
        Use Verlet integral method to do the simulation, return the trajectory in the form of
        traj[time_slice, particle_index, dim]
        """
        vtk_writer = vtktools.VTK_XML_Serial_Unstructured()
        
        if not self.initialized:
            # create random points inthe box
            self.traj[0] = np.random.rand(self.par_num, self.dim) * self.box
            v0 = np.zeros((self.par_num, self.dim), dtype=float)

        self.traj[1] = self.traj[0] + self.v0
        next = 2
        while next < self.traj.shape[0]:
            force = LJ_force(self.traj[1])
            self.traj[next] = 2*self.traj[next-1] - self.traj[next-2] + self.dt**2*force
            # periodic boundary        
            self.traj[next] = np.mod(self.traj[next], self.box)
            #vktools
            r_x = self.traj[next][:,0]
            r_y = self.traj[next][:,1]
            r_z = self.traj[next][:,2]
            F_x = force[:,0]
            F_y = force[:,1]
            F_z = force[:,2]
            vtk_writer.snapshot("simu/MD_"+ str(self.dt) + ".vtu", r_x,r_y,r_z,
                x_force = F_x, y_force = F_y, z_force = F_z)

            next += 1

        vtk_writer.writePVD("MD.pvd")
        return self.traj


def LJ(r, direction):
    return 4 * (-12*r**(-13) + 6*r**(-7)) * direction


def LJ_force(pos_all):
    """
    Lennard-Jones potential
    """
    par_num = pos_all.shape[0]

    
    force = np.zeros(pos_all.shape)
    for par_index1 in range(par_num):
        for par_index2 in range(par_num):
            if par_index1 != par_index2:
                pos1 = pos_all[par_index1]
                pos2 = pos_all[par_index2]
                dist = pos1-pos2
                r = np.linalg.norm(dist)
                force[par_index1] = -LJ(r, dist/r)
                # print(force)

    return force
    

# np.random.seed(10)
test = particle_dynamics_sim(dt=0.1, iter_num=1000, par_num = 10, dim = 3, box = (10.,10.,10.))
result = test.run()