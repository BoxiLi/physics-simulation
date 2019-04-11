from mc_simulation import *


class Metropolis_Ising_sim(discrete_Ising_sim):
    def __init__(self, size, J, h, kT, iter_num, thermal_num):
        discrete_Ising_sim.__init__(self, size, J, h, iter_num, thermal_num)
        if kT <= 0:
            raise ValueError("Temperature must be non negative")
        elif kT == 0.:
            self.kT = 1.0e-17
        else:
            self.kT = kT
        self.energy = self.calculate_energy()
        self.exp_dict = {}


    def get_energy(self):
        return self.energy


    def calculate_energy_diff(self, origin_pos):
        """
        calculate the energy difference if the spin at pos is flipped E_flipped - E_old
        pos is list of integer indicates the index of the current spin
        """
        energy_diff = 0
        s0 = self.lattice[tuple(origin_pos)] 
        s0 = np.sum(s0) # remove unnecenssary braket, it's only an integer
        for neighbour_axis in range(self.dim): # a loop over all directions
            # in each dimension a spin has two direction
            neighb1_pos = list(origin_pos)
            temp1 = np.mod(origin_pos[neighbour_axis] - 1, self.size[neighbour_axis])
            neighb1_pos[neighbour_axis] = temp1
            s1 = self.lattice[tuple(neighb1_pos)] # IMPORTANT to use TUPLE! List is NOT the same!
            s1 = np.sum(s1)

            neighb2_pos = list(origin_pos)
            temp2 = np.mod(origin_pos[neighbour_axis] + 1, self.size[neighbour_axis])
            neighb2_pos[neighbour_axis] = temp2
            s2 = self.lattice[tuple(neighb2_pos)]
            s2 = np.sum(s2)

            energy_diff += -2.*self.J*s1*(-1.)*s0 - 2.*self.J*s2*(-1.)*s0

        energy_diff += -2.*self.h*(-1.)*s0
        return energy_diff


    def update_flip(self, pos):
        """
        One metropolis step, flip the spin according to a certain state
        """
        if not isinstance(pos, tuple):
            pos = tuple(pos)
        energy_diff =  self.calculate_energy_diff(pos)
        if energy_diff < 0:
            self.lattice[pos] = -self.lattice[pos]
            self.energy += energy_diff
        else:
            try:
                self.exp_dict[energy_diff]
            except:
                self.exp_dict[energy_diff] = np.exp(-energy_diff/self.kT)
                
            R = np.random.rand()
            if R < self.exp_dict[energy_diff]:
                self.lattice[pos] = -self.lattice[pos]
                self.energy += energy_diff
        
    
    def thermalize(self, new_thermal_num = None):
        """
        Run many metropolis steps but not record the result
        """
        # If a new thermalization loop number is given, replace the default value with it
        if new_thermal_num:
            if isinstance(new_thermal_num, int):
                self.thermal_num = new_thermal_num
            else:
                raise ValueError("Thermalization loop number has to be an integer")
        # Generate the random matrix
        rand_pos = np.empty([self.thermal_num, self.dim], dtype = np.int)
        for axis in range(self.dim):
            rand_pos[:,axis] = np.random.randint(0, self.size[axis], self.thermal_num)
        # Thermalization
        step = 0
        while(step < self.thermal_num):
            pos = tuple(rand_pos[step])
            self.update_flip(pos) 
            step += 1


    def run(self, new_iter_num = None, gap = 0):
        """
        Run the whole simulation, including thermalization.
        """
        # Thermalize
        self.thermalize()
        # If a new simulation loop number is given, replace the default value with it
        if new_iter_num:
            if isinstance(new_iter_num, int):
                self.iter_num = new_iter_num
            else:
                raise ValueError("Metropolis loop number has to be an integer")
        # Generate the random matrix
        rand_pos = np.empty([self.iter_num, self.dim], dtype = np.int)
        for axis in range(self.dim):
            rand_pos[:,axis] = np.random.randint(0, self.size[axis], self.iter_num) 
        # Simulation
        step = 0
        gap_counter = 0
        while(step < self.iter_num):
            pos = tuple(rand_pos[step])
            self.update_flip(pos)
            gap_counter += 1
            if (gap_counter >= gap):
                try:
                    self.simulation_data["energy"].append(self.get_energy())
                    self.simulation_data["m"].append(np.sum(self.lattice))
                except(KeyError):
                    self.simulation_data["energy"] = [self.get_energy()]
                    self.simulation_data["m"] = [np.sum(self.lattice)]
                # result_data["lattice"].append(lattice.copy())
                step += 1
                gap_counter = 0
        return self.simulation_data