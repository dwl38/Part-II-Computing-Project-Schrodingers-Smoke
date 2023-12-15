import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.optimize import curve_fit
from ..common import round_sig
from ..ssmoke2d import SSmoke2D

#====================================================================================================
# Testing the time complexity of the SSmoke2D integrator.
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nmax = 450
Nmin = 10
Npts = 30
Ntimes = 200

#----------------------------------------------------------------------------------------------------
# Set-up; also checks to see if there is already pre-existing data, and includes it if so.
#----------------------------------------------------------------------------------------------------

rng = np.random.default_rng()
numbers = []
intervals = []

if os.path.isfile('prerendered\\TimingTest.npz'):
    data = np.load('prerendered\\TimingTest.npz')
    for i in range(len(data['n'])):
        numbers.append(data['n'][i])
        intervals.append(data['t'][i])

def fitting_linear(x, coeff):
    return coeff * x

def fitting_log(x, coeff):
    return coeff * x * np.log2(x)

dx = 0.05
dt = 0.02
hbar = 0.02

def initial_vel_field(pos):
    x, y = pos
    return (-y, x)

#----------------------------------------------------------------------------------------------------
# Test
#----------------------------------------------------------------------------------------------------

print()
print(r'Starting timing test...')

for _ in range(Npts):

    Nx = rng.integers(Nmin, Nmax)
    Ny = rng.integers(Nmin, Nmax)

    integrator = SSmoke2D((Nx, Ny), (dx, dx), hbar, periodic=True, initValues=initial_vel_field)
    start = time.time()
    for _ in range(Ntimes):
        integrator.advance_timestep(dt)
    end = time.time()
    interval = 1000 * (end - start) / Ntimes

    print(f'    Nx = {Nx}, Ny = {Ny}, NxNy = {Nx * Ny}, time = {interval} ms.')
    numbers.append(Nx * Ny)
    intervals.append(interval)

print()
print(r'Testing complete.')

#----------------------------------------------------------------------------------------------------
# Saving data
#----------------------------------------------------------------------------------------------------

print(r'Saving...')

np.savez('prerendered\\TimingTest.npz', n=numbers, t=intervals)

print(r'Saving complete.')
print()

#----------------------------------------------------------------------------------------------------
# Plotting data, including best-fit lines
#----------------------------------------------------------------------------------------------------

numbers, intervals = zip(*sorted(zip(numbers, intervals)))
numbers = np.array(numbers, dtype=int)
intervals = np.array(intervals, dtype=float)

popt_lin, pcov_lin = curve_fit(fitting_linear, numbers, intervals)
popt_log, pcov_log = curve_fit(fitting_log, numbers, intervals)
perr_lin = np.sqrt(np.diag(pcov_lin))[0]
perr_log = np.sqrt(np.diag(pcov_log))[0]

print(u'Best fit linear:  coeff = {} \u00b1 {} ({}%).'.format(round_sig(popt_lin[0]), round_sig(perr_lin, 2), round_sig(100 * perr_lin / popt_lin[0], 2)))
print(u'Best fit lin-log: coeff = {} \u00b1 {} ({}%).'.format(round_sig(popt_log[0]), round_sig(perr_log, 2), round_sig(100 * perr_log / popt_log[0], 2)))
print()
print(r'Displaying data and then exiting.')
print()

plt.loglog(numbers, fitting_linear(numbers, *popt_lin), 'k-', label=r'$O(n)$ fit')
plt.loglog(numbers, fitting_log(numbers, *popt_log), 'b-', label=r'$O(n \log n)$ fit')
plt.loglog(numbers, intervals, 'rx', label=r'Data')
plt.xlabel(r'$n = N_xN_y$')
plt.ylabel(r'Time (ms)')
plt.legend()
plt.show()




