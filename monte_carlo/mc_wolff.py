from mc_simulation import *


class wolff_ising_sim(discrete_ising_sim):
    def __init__(self, size, J, h, kT, iter_num, thermal_num):
        discrete_ising_sim.__init__(self, size, J, h, iter_num, thermal_num)
        if kT <= 0:
            raise ValueError("Temperature must be non negative")
        elif kT == 0.:
            self.kT = 1.0e-17
        else:
            self.kT = kT
        self.p = 1-np.exp(-2*J/kT)
        

    def wolff_update(self, root):
        """
        Start from the spin at root, go through all the spins with same sites and 
        flip it with probability p
        root is a tuple denoting the position of the starting spin
        Return the size of the found cluster
        """
        if not isinstance(root, tuple):
            root = tuple(root)
        queue = [root]
        cluster = {root}
        root_spin = self.lattice[root]
        while queue: # queue is not empty
            # get a list of all new neighbours' position
            pos = queue.pop(0)
            neighbours_set = self.get_neighbours(pos)
            new_neighbours_set = neighbours_set - cluster # removed already fliped ones
            # flip the spin according to the update rule
            for new_neighbour in new_neighbours_set:
                '''
                The neighbours will only be added if it has the same spin and it has never been 
                assigned to any group (not discovered). 
                '''
                if self.lattice[new_neighbour] == root_spin:
                    if np.random.random() < self.p:
                        queue.append(new_neighbour)
                        cluster.add(new_neighbour)
                        self.lattice[new_neighbour] = -self.lattice[new_neighbour]
        return len(cluster)


    def thermalize(self, new_thermal_num = None):
        """
        Run many wolff steps but not record the result
        """
        # If a new thermalization loop number is given, replace the default value with it
        discrete_ising_sim.thermalize(self, self.wolff_update, new_thermal_num = None)


    def run(self, new_iter_num = None, gap = 0):
        """
        Run the whole simulation, including thermalization.
        Return a dictionary of the measured values
        """
        # Thermalize
        discrete_ising_sim.thermalize(self, update_method = self.wolff_update)
        # If a new simulation loop number is given, replace the default value with it
        if new_iter_num:
            if isinstance(new_iter_num, int):
                self.iter_num = new_iter_num
            else:
                raise ValueError("Wolff loop number has to be an integer")
        # Generate the random matrix
        rand_pos = np.empty([self.iter_num, self.dim], dtype = np.int)
        for axis in range(self.dim):
            rand_pos[:,axis] = np.random.randint(0, self.size[axis], self.iter_num) 
        # Simulation
        step = 0
        gap_counter = 0
        while(step < self.iter_num):
            pos = tuple(rand_pos[step])
            cluster_size = self.wolff_update(pos)
            gap_counter += 1
            if (gap_counter >= gap):
                try:
                    self.simulation_data["energy"].append(self.get_energy())
                    self.simulation_data["m"].append(np.sum(self.lattice))
                    self.simulation_data["cluster_size"].append(cluster_size)
                except(KeyError):
                    self.simulation_data["energy"] = [self.get_energy()]
                    self.simulation_data["m"] = [np.sum(self.lattice)]
                    self.simulation_data["cluster_size"] = [cluster_size]
                # result_data["lattice"].append(lattice.copy())
                step += 1
                gap_counter = 0
        return self.simulation_data


# dim = 2
# size = [10]*dim
# site_num = np.prod(size)
# J = 1.
# h = 0.
# kT = 2
# iter_num=0
# thermal_num=0
# test = wolff_ising_sim(size, J, h, kT, iter_num, thermal_num)
# test.thermalize(1000)
# a = np.random.randint(0,10,2)
# test.wolff_update(a)