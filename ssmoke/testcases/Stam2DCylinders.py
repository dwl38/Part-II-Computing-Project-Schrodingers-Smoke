from ..common import *
from ..stam2d import Stam2D
from ..dataio import save_2d

#====================================================================================================
# Initial condition: 'circles' of fluid moving left and right
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nx = 400
Ny = 200
Nframes = 125
steps_per_frame = 10
Nsteps = Nframes * steps_per_frame

dx = 0.05
dy = 0.05
dt = 0.02
Lx = Nx * dx
Ly = Ny * dy

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

Cx1 = 0.25 * Lx
Cx2 = 0.75 * Lx
Cy = 0.5 * Ly
rad = min(0.5 * Cx1, 0.5 * Cy)

def initial_vel_field(pos):
    x, y = pos
    if ((x - Cx1)**2 + (y - Cy)**2) < rad**2:
        return (1.0, 0.0)
    elif ((x - Cx2)**2 + (y - Cy)**2) < rad**2:
        return (-1.0, 0.0)
    return (0.0, 0.0)

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
save_2d('prerendered\\Stam2DCylinders.data', Nframes, (Nx, Ny), (dx, dy), times, xcoords, ycoords, vx, vy)
print('Saving complete.')


