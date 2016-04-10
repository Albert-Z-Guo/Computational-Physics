from __future__ import division
from random import random, randint
import numpy as np

def ising2d_simulation(temp_start=1, temp_end=4, temp_step=0.1, mcsteps=1000, L=10):

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
            spin = np.ones((L, L))
        return spin

    def compute_mcsteps(spin, temperature, mcsteps, L):
        """
        We can't combine the monte carlo step iterations and measurement loop
        into one single loop when we are randomly picking points on the grid.
        This is a slight variant of the algorithm given on pg 247 of G&N.
        """
        energy, energy_squared, magnetization = 0, 0, 0
        for one_mcstep in xrange(mcsteps):
            # each mcstep performs L**2 updates on the square spin lattice.
            for spin_site in xrange(L**2):
                # Randomly choose a site on the grid.
                x, y = randint(0, L-1), randint(0, L-1)
                # Get neighbor indices, accounting for periodic boundaries.
                x_prev = L-1 if x==0   else x-1
                x_next = 0   if x==L-1 else x+1
                y_prev = L-1 if y==0   else y-1
                y_next = 0   if y==L-1 else y+1
                # Calculate the potential change in energy.
                delta_energy = 2*spin[x, y]*(spin[x_prev, y] + spin[x_next, y] +
                                                spin[x, y_prev] + spin[x, y_next])
                # Spin flip condition, the exp term is the Boltzmann factor.
                if delta_energy<=0 or np.exp(-delta_energy/temperature)>random():
                    spin[x, y] = -spin[x, y]
            # accumulate energy and magnetization through the chosen monte carlo steps by simpler vectorization
            E, M = 0, 0
            M += np.sum(spin)
            magnetization += M
            left_spin = np.roll(spin, 1, axis=0)
            lower_spin = np.roll(spin, 1, axis=1)
            E -= np.sum(spin*(left_spin + lower_spin))
            energy += E
            energy_squared += np.square(E)
        
        '''
        # you may check the following variables by printing
        print 'E  = %d' %(energy)
        print '<E>  = %f ' %(energy/(mcsteps*L**2))
        print 'E^2  = %d' %(energy_squared)
        print '<E^2>  = %f' %(energy_squared/(mcsteps*L**4))
        '''

        return (spin, abs(magnetization), energy, energy_squared)

    # initialize the spin lattice and observables.
    spin = initialize_spins(L)
    temp_range = np.arange(temp_start, temp_end + temp_step, temp_step)
    mean_magnetization = np.zeros_like(temp_range)
    mean_energy = np.zeros_like(temp_range)
    energy_variance = np.zeros_like(temp_range)
    specfic_heat = np.zeros_like(temp_range)

    # run simulation (magnetization and energy aren't saved during thermalization).
    print ('Metropolis algorthm %d Monte Carlo Steps of %dx%d spin site' %(mcsteps, L, L))
    print 'Starting thermalization cycle ...'
    spin, magnetization, energy, energy_squared = compute_mcsteps(spin, temp_start, mcsteps, L)
    print 'Starting measurement cycle ...'
    for i, temperature in enumerate(temp_range):
        spin, magnetization, energy, energy_squared = compute_mcsteps(spin, temperature, mcsteps, L)
        mean_magnetization[i] =  magnetization/(mcsteps*L**2)
        mean_energy[i] = energy/(mcsteps*L**2)
        energy_variance[i] = energy_squared/(mcsteps*L**4) - np.square(mean_energy[i])
        specfic_heat[i] = energy_variance[i]/np.square(temperature)
        print ('temperature, mean_magnetization, mean_energy = %.3f, %.4f, %.4f, %.4f' %  
                (temperature, mean_magnetization[i], mean_energy[i], specfic_heat[i]))

    return (spin, temp_range, mean_magnetization, mean_energy, specfic_heat, mcsteps, L)

def plot_spin_lattice(spin, temperature):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.5, 5))
    ax = fig.add_subplot(111)
    ax.set_title('$\\rm{\\bf Spin\,Lattice:}\,T=%s\,K$' % (temperature),
        fontsize=14, loc=('center'))
    ax.imshow(spin, origin='lower', interpolation='none')
    plt.tight_layout()
    plt.show()

def plot_observables(temp_range, mean_magnetization, mean_energy, specfic_heat, mcsteps, L):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4.5))
    """
    ax = fig.add_subplot(121, xlim=(min(temp_range), max(temp_range)),
        ylim=(0, max(mean_magnetization)))
    ax.set_xlabel('$\\rm Temperature$', fontsize=14)
    ax.set_ylabel('$\\rm Magnetization$', fontsize=14)
    ax.set_title('$\\rm{\\bf Ising\,2D:}\,%s ^2 Grid,\,%s\,MCSteps$' % (L, mcsteps),
        fontsize=14, loc=('center'))
    ax.plot(temp_range, mean_magnetization, 'o', markersize=6, color='b')
    """
    ax = fig.add_subplot(121, xlim=(0, 5),
        ylim=(min(mean_energy), max(mean_energy)))
    ax.set_xlabel('$\\rm Temperature$', fontsize=14)
    ax.set_ylabel('$\\rm Energy$', fontsize=14)
    ax.set_title('$\\rm{\\bf Ising\,2D:}\,%s ^2 Grid,\,%s\,Monte\,Carlo\,steps$' % (L, mcsteps),
        fontsize=14, loc=('center'))
    ax.plot(temp_range, mean_energy, 'o', markersize=6, color='r')
    
    ax = fig.add_subplot(122, xlim=(0, 5), 
        ylim=(min(specfic_heat), max(specfic_heat)))
    ax.set_xlabel('$\\rm Temperature$', fontsize=14)
    ax.set_ylabel('$\\rm Specific\,heat\,per\,spin$', fontsize=14)
    ax.set_title('$\\rm{\\bf Ising\,2D:}\,%s ^2 Grid,\,%s\,Monte\,Carlo\,steps$' 
        % (L, mcsteps), fontsize=14, loc=('center'))
    ax.plot(temp_range, specfic_heat, 'o', markersize=6, color='b')

    plt.tight_layout()
    plt.show()

def main():
    spin, temp_range, mean_magnetization, mean_energy, specfic_heat, mcsteps, L = ising2d_simulation()
    plot_observables(temp_range, mean_magnetization, mean_energy, specfic_heat, mcsteps, L)
    plot_spin_lattice(spin, temp_range[-1])

if __name__ == '__main__':
    main()
