from __future__ import division
from random import random, randint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pickle
from sys import setrecursionlimit # set recursion limit above the system default
setrecursionlimit(10000)

# define key constants
L = 10 # length of the spin site
monte_carlo_steps = 1000 # number of monte carlo steps performed (required scaling as ~4*L).
temperature_start = 1 # [K]
temperature_end = 4 # [K]
temperature_step = 0.1
store_data = 1 # (0/1) boolean variable to indicate whether to store data or not

# initialize the spin sites and return the spin sites
def initialize_spins(L, randomize_spins=False):
    """
    randomize_spins==True: set each site to +1 or -1.
    randomize_spins==False: set all spins to +1.
    """
    if randomize_spins:
        # use this option when starting above temperatures of ~ 2.
        spin = np.random.random_integers(0, 1, (L, L))
        spin[spin==0] = -1
    else:
        # use this option when starting below temperatures of ~ 2.
        spin = np.ones((L, L, 1))
    return spin

# choose a random spin in site, grow a cluster, flip the cluster, and return the updated spin sites
def one_wolff_monte_carlo_step(temperature, L, spin, probability_add):
    """
    One Wolff Monte Carlo step:
    1. At each Monte Carlo step, a single cluster is grown 
    around a randomly chosen seed spin.
    2. All of the spins in this cluster are then flipped.
    """
    
    def consider_adding_to_cluster(x, y, spin, cluster, temperature, clusterspin):
        """
        Adding to cluster:
        A neighbor spin is added to the cluser 
        1. if it not only has the same spin as the chosen seed spin
        2. but also satisfies the Boltzmann Criterion
        """
        # a neighbor spin will be added to the cluster 
        # only if it is aligned with (both +/-) the seed spin
        if spin[x, y] == clusterspin:
            if random() < probability_add:
                spin = grow_cluster(temperature, x, y, cluster, spin)
        return spin

    def grow_cluster(termperature, x, y, cluster, spin):
        """
        Growing a Wolff cluster:
        1. The spin is marked as belonging to the cluster.
        2. The four nearest neighbors are checked one by one.
        If the neighbor does not belong to the cluster, 
        consider added it to the cluster.
        3. When the cluster stops growing, flip the whole cluster.
        """
        # mark the spin as belonging to the cluster
        cluster[x, y] = True

        # assume periodic boundary
        x_prev = L-1 if x==0   else x-1
        x_next = 0   if x==L-1 else x+1
        y_prev = L-1 if y==0   else y-1
        y_next = 0   if y==L-1 else y+1

        # if a neighbor spin does not belong to the cluster, consider adding it to the cluster
        # clusterspin holds the value (+/-1) of the seed spin
        if(cluster[x_prev, y] == 0): spin = consider_adding_to_cluster(x_prev, y, spin, cluster, temperature, clusterspin = spin[x,y])
        if(cluster[x_next, y] == 0): spin = consider_adding_to_cluster(x_next, y, spin, cluster, temperature, clusterspin = spin[x,y])
        if(cluster[x, y_prev] == 0): spin = consider_adding_to_cluster(x, y_prev, spin, cluster, temperature, clusterspin = spin[x,y])
        if(cluster[x, y_next] == 0): spin = consider_adding_to_cluster(x, y_next, spin, cluster, temperature, clusterspin = spin[x,y])

        # return the spin sites when the cluster stops growing
        return spin

    # define a cluster labeling array
    # no cluster is defined yet; clear the cluster array
    cluster = np.zeros((L, L), dtype=bool) 
    # choose a random spin
    x, y = randint(0, L-1), randint(0, L-1)
    # grow a cluster
    spin = grow_cluster(temperature, x, y, cluster, spin)
    # flip the whole cluster
    spin[cluster] *= -1
    # return the updated spin sites
    return spin


def ising_model_simulation(spin, temperature, L, monte_carlo_steps):
    
    # define Boltzmann Criterion
    probability_add = 1 - np.exp(-2/temperature)
    
    # ready to accumulate energy through the chosen monte carlo steps
    energy = 0
    energy_squared = 0

    # perform the chosen monte carlo steps
    for iteration_number in xrange(monte_carlo_steps):
        # reset the summing ammount of energy in each monte_carlo_step
        E = 0
        spin = one_wolff_monte_carlo_step(temperature, L, spin, probability_add)
        # accumulate energy through the chosen monte carlo steps by vectorization
        left_spin = np.roll(spin, 1, axis=0)
        lower_spin = np.roll(spin, 1, axis=1)
        E = -np.sum(spin*(left_spin + lower_spin))
        energy += E
        energy_squared += np.square(E)
    """
    # you may check the following variables by printing
    print 'E  = %d' %(energy)
    print '<E>  = %f ' %(energy/(monte_carlo_steps*L**2))
    print 'E^2  = %d' %(energy_squared)
    print '<E^2>  = %f' %(energy_squared/(monte_carlo_steps*L**4))
    """
    return (spin, energy, energy_squared)


def plot_spin_lattice(spin_array, temperature_range):
    # slider update function
    def update(val):
        i = int(val)
        ax.set_title('$\\rm{\\bf Spin\,Lattice:}\;T= %.3f\,\\rm [K]$'
            % (temperature_range[i]), fontsize=14, loc=('center'))
        im.set_array(spin_array[:,:,i])
    
    # plot the spin site
    fig = plt.figure(figsize=(4.5, 5))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.12, top=1)
    ax.set_title('$\\rm{\\bf Spin\,Lattice:}\;T= %.3f\,\\rm [K]$' 
        % (temperature_range[-1]), fontsize=14, loc=('center'))
    im = ax.imshow(spin_array[:,:,-1], origin='lower', interpolation='none')
    slider_ax = plt.axes([0.2, 0.06, 0.6, 0.03], axisbg='#7F0000')
    spin_slider = Slider(slider_ax, '', 0, len(temperature_range)-1, len(temperature_range)-1, 
        valfmt ='%u', facecolor='#00007F')
    spin_slider.on_changed(update)
    plt.annotate('Temperature Slider', xy=(0.32,0.025), xycoords='figure fraction', 
        fontsize=12)
    
    plt.show()


def plot_measurement(temperature_range, mean_energy, specfic_heat, monte_carlo_steps, L):
    fig = plt.figure(figsize=(12, 5))
    # subplot for energy vs. temperature
    ax = fig.add_subplot(121, xlim=(0, 5), 
        ylim=(min(mean_energy), max(mean_energy)))
    ax.set_xlabel('$\\rm Temperature$', fontsize=14)
    ax.set_ylabel('$\\rm Energy\,per\,spin$', fontsize=14)
    ax.set_title('$\\rm{\\bf Ising\,2D:}\,%s ^2 Grid,\,%s\,Monte\,Carlo\,steps$' 
        % (L, monte_carlo_steps), fontsize=14, loc=('center'))
    ax.plot(temperature_range, mean_energy, 'o', markersize=6, color='r')
    
    # subplot for specific heat vs. temperature
    ax = fig.add_subplot(122, xlim=(0, 5), 
        ylim=(min(specfic_heat), max(specfic_heat)))
    ax.set_xlabel('$\\rm Temperature$', fontsize=14)
    ax.set_ylabel('$\\rm Specific\,heat\,per\,spin$', fontsize=14)
    ax.set_title('$\\rm{\\bf Ising\,2D:}\,%s ^2 Grid,\,%s\,Monte\,Carlo\,steps$' 
        % (L, monte_carlo_steps), fontsize=14, loc=('center'))
    ax.plot(temperature_range, specfic_heat, 'o', markersize=6, color='b')

    plt.show()


def main():
    # define temperature range for the measurement
    temperature_range = np.arange(temperature_start, temperature_end + temperature_step, temperature_step)
    
    # define measurement variables in the temperature range
    mean_energy = np.zeros_like(temperature_range)
    energy_variance = np.zeros_like(temperature_range)
    specfic_heat = np.zeros_like(temperature_range)
    susceptibility = np.zeros_like(temperature_range)
    spin_array = np.tile(initialize_spins(L), (1, 1, len(temperature_range))) # store all the measurements to an array for the later slider display
    
    # indicate the monte carlo steps used in the simulation
    print ('Wolff algorithm %d Monte Carlo Steps of %dx%d spin site' %(monte_carlo_steps, L, L))
    
    # thermalize at temperature_start
    print ('Starting thermalization cycle ...')
    spin_array[:, :, 0], energy, energy_squared = ising_model_simulation(spin_array[:, :, 0], temperature_start, L, monte_carlo_steps)
    
    # take measurements across different temperatures
    print ('Starting measurement cycle ...')
    for i, temperature in enumerate(temperature_range):
        spin_array[:, :, i], energy, energy_squared = ising_model_simulation(spin_array[:, :, i], temperature, L, monte_carlo_steps)
        mean_energy[i] = energy/(monte_carlo_steps*L**2)
        energy_variance[i] = energy_squared/(monte_carlo_steps*L**4) - np.square(mean_energy[i])
        specfic_heat[i] = energy_variance[i]/np.square(temperature)
        print ('temperature, mean_energy, specific_heat = %.3f, %.4f, %.4f' % (temperature, mean_energy[i], specfic_heat[i]))

    if store_data == 1:
        # store variabls in pickle format for future use
        output = open('10x10e1.pkl', 'wb')
        pickle.dump(mean_energy, output)
        output.close()
        output = open('10x10s1.pkl', 'wb')
        pickle.dump(spin_array, output)
        output.close()
        output = open('10x10c1.pkl', 'wb')
        pickle.dump(specfic_heat, output)
        output.close()
    
    # plot spin lattice at temperature_end and measurement 
    plot_spin_lattice(spin_array, temperature_range)
    plot_measurement(temperature_range, mean_energy, specfic_heat, monte_carlo_steps, L)

if __name__ == '__main__':
    main()


