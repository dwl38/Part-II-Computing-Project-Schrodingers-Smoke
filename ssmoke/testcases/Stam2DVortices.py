from math import sqrt
from ..common import *
from ..stam2d import Stam2D
from ..dataio import save_2d

#====================================================================================================
# Initial condition: two counter-rotating vortices
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nx = 400
Ny = 200
Nframes = 50
steps_per_frame = 2
Nsteps = Nframes * steps_per_frame

dx = 0.005
dy = 0.005
dt = 0.01
Lx = Nx * dx
Ly = Ny * dy
rmin = sqrt(16 * dx * dy)

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

def vortex(x, y, x0, y0, vort): # Velocity function for vortex at (x0, y0)
    rsq = (x - x0)**2 + (y - y0)**2
    rminsq = rmin**2
    if rsq > rminsq:
        return (vort * (y0 - y) / rsq, vort * (x - x0) / rsq, -vort**2 / (2 * rsq))
    return (vort * (y0 - y) / rminsq, vort * (x - x0) / rminsq, vort**2 * ((rsq / (2 * rminsq)) - 1) / rminsq)

def initial_vel_field(pos):
    x, y = pos
    n_sum = 3
    vx = 0
    vy = 0
    
    for i in range(-n_sum, n_sum+1):
        for j in range(-n_sum, n_sum+1):
            left = vortex(x, y, (i + 0.25)*Lx, (j + 0.5)*Ly, 0.9 * rmin)
            right = vortex(x, y, (i + 0.75)*Lx, (j + 0.5)*Ly, -0.9 * rmin)
            vx += left[0] + right[0]
            vy += left[1] + right[1]

    return (vx, vy)

integrator = Stam2D((Nx, Ny), (dx, dy), periodic=True, initValues=initial_vel_field)
xcoords, ycoords = integrator.meshgrid()
vx = [None] * Nframes
vy = [None] * Nframes
times = [None] * Nframes
vx[0], vy[0] = integrator.flow_vel()
times[0] = 0.0

#----------------------------------------------------------------------------------------------------
# Simulation
#----------------------------------------------------------------------------------------------------

print()
print('Calculating flow via Stam2D...')
for frame in range(1, Nframes):
    for t in range(steps_per_frame):
        print_progress_bar(frame * steps_per_frame + t, 0, Nsteps)
        integrator.advance_timestep(dt)
    vx[frame], vy[frame] = integrator.flow_vel()
    times[frame] = round_sig(frame * steps_per_frame * dt)
print()
print('Calculation complete.')
print()

#----------------------------------------------------------------------------------------------------
# Saving data & creating animation
#----------------------------------------------------------------------------------------------------

print('Saving data...')
save_2d('prerendered\\Stam2DVortices.data', Nframes, (Nx, Ny), (dx, dy), times, xcoords, ycoords, vx, vy)
print('Saving complete.')


