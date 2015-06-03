import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from wolff import temperature_start, temperature_end, temperature_step # import variables

# minor modification on the title of the orignial plotting function
def plot_measurement(temperature_range, mean_energy, specfic_heat):
    fig = plt.figure(figsize=(12, 5))
    # subplot for energy vs. temperature
    ax = fig.add_subplot(121, xlim=(0, 5), 
        ylim=(min(mean_energy), max(mean_energy)))
    ax.set_xlabel('$\\rm Temperature$', fontsize=14)
    ax.set_ylabel('$\\rm Energy\,per\,spin$', fontsize=14)
    ax.set_title('$\\rm{\\bf Ising\,2D:}\,Multiple\,Grids,\,Multiple\,Monte\,Carlo\,steps$', 
    fontsize=14, loc=('center'))
    ax.plot(temperature_range, mean_energy, 'o', markersize=6, color='r')
    
    # subplot for specific heat vs. temperature
    ax = fig.add_subplot(122, xlim=(0, 5), 
        ylim=(min(specfic_heat), max(specfic_heat)))
    ax.set_xlabel('$\\rm Temperature$', fontsize=14)
    ax.set_ylabel('$\\rm Specific\,heat\,per\,spin$', fontsize=14)
    ax.set_title('$\\rm{\\bf Ising\,2D:}\,Multiple\,Grids,\,Multiple\,Monte\,Carlo\,steps$', 
        fontsize=14, loc=('center'))
    ax.plot(temperature_range, specfic_heat, 'o', markersize=6, color='b')
    plt.show()

temperature_range = np.arange(temperature_start, temperature_end + temperature_step, temperature_step)

# import stored variables
pkl_file = open('5x5c.pkl', 'rb')
specfic_heat1 = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('10x10c.pkl', 'rb')
specfic_heat2 = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('20x20c.pkl', 'rb')
specfic_heat3 = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('32x32c.pkl', 'rb')
specfic_heat4 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('5x5e.pkl', 'rb')
mean_energy1 = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('10x10e.pkl', 'rb')
mean_energy2 = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('20x20e.pkl', 'rb')
mean_energy3 = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('32x32e.pkl', 'rb')
mean_energy4 = pickle.load(pkl_file)
pkl_file.close()

# concatenate the arrays
temperature_range = np.tile(temperature_range, 4)
specfic_heat = np.hstack((specfic_heat1, specfic_heat2, specfic_heat3, specfic_heat4))
mean_energy = np.hstack((mean_energy1, mean_energy2, mean_energy3, mean_energy4))

# plot measurements
plot_measurement(temperature_range, mean_energy, specfic_heat)



