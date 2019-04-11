import numpy as np
import matplotlib.pyplot as plt

class discrete_Ising_sim(object):
    def __init__(self, size, J, h, iter_num, thermal_num):
        self.size = tuple(size)
        self.dim = len(size)
        self.J = J
        self.h = h
        self.iter_num = iter_num
        self.thermal_num = thermal_num
        self.lattice = np.random.randint(0, 2, size, int) * 2 - 1
        self.simulation_data = {}

    
    def get_lattice(self):
        return self.lattice


    def get_energy(self):
        return self.calculate_energy()


    def calculate_energy(self):
        """
        Given a lattice, calculate the total energy H = -J*s1*s2 - h*s
        """
        energy = 0.

        # choose one axis to iterate and calculate the energy of all the neighbour-pairs
        # in this direction. The hypersurface orthogonal to this axis is multiplied with
        # the next layer.
        for iter_axis in range(self.dim):
            all_keep_slice = slice(0, self.size[iter_axis]) # slice object that keeps everything
            # print(iter_axis,"\n")
            slice_layer_list = []

            # generate a list of the slice object, e.g. for iter_axis=2 and i=3, it will 
            # be [:,:,3,:]
            for i in range(self.size[iter_axis]):
                slice_layer = [all_keep_slice] * self.dim
                slice_layer[iter_axis] = slice(i,i+1)
                slice_layer_list.append(tuple(slice_layer))

            # calculate the multiplication and summation
            for i in range(self.size[iter_axis]):
                j = np.mod(i+1, self.size[iter_axis])
                temp = np.multiply(self.lattice[slice_layer_list[i]], self.lattice[slice_layer_list[j]])
                energy -= self.J * np.sum(temp)

        energy -= self.h * np.sum(self.lattice)
        return energy


    def get_neighbours(self, origin_pos):
        """
        This is a version that work in N dimensions. It returns a list of all the neighbours'
        coordinates
        origin_pos: a tuple corresponds to the position of current spin
        """
        neighbours_list = []
        for neighbour_axis in range(self.dim):
            neighb1_pos = list(origin_pos)
            neighb1_pos[neighbour_axis] = np.mod(origin_pos[neighbour_axis] - 1, self.size[neighbour_axis])
            neighb1_pos = tuple(neighb1_pos)  # IMPORTANT to use TUPLE! list is NOT the same!
            neighbours_list.append(neighb1_pos)

            neighb2_pos = list(origin_pos)
            neighb2_pos[neighbour_axis] = np.mod(origin_pos[neighbour_axis] + 1, self.size[neighbour_axis])
            neighb2_pos = tuple(neighb2_pos) 
            neighbours_list.append(neighb2_pos)
        return neighbours_list


def autocorrelation(x):
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal. Tell them this!!
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    ac = np.real(pi)[:x.size // 2] / np.sum(xp**2)

    taucorr = np.argmax(ac < 0) # argmax([F,F,F,F,F,F,T,T,F,T,F,T]), returns the index for the first T
    taucorr = 0.5 + 2. * np.sum(ac[0:taucorr])
    sigma = np.var(x) * taucorr / float(len(x))
    sigma = np.sqrt(sigma)
    return np.mean(x), sigma, taucorr