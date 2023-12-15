import numpy as np
from ..common import *
from ..ssmoke2d import SSmoke2D, VelocityConstraint2D
from ..dataio import save_2d

#====================================================================================================
# Initial condition: fluid moving around a stationary cylinder
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nx = 500
Ny = 250
Nframes = 200
steps_per_frame = 10
Nsteps = Nframes * steps_per_frame

dx = 0.004
dy = 0.004
dt = 0.002
hbar = 0.002
Lx = Nx * dx
Ly = Ny * dy

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

Cx = 0.25 * Lx
Cy = 0.5 * Ly
rad = 0.125 * Ly

def initial_vel_field(pos):
    x, y = pos
    if ((x - Cx)**2 + (y - Cy)**2) < rad**2:
        return (0.0, 0.0)
    return (1.0, 0.0)

region = np.zeros((Nx, Ny), dtype=bool)
for i in range(Nx):
    for j in range(Ny):
        if (((i * dx) - Cx)**2 + ((j * dy) - Cy)**2) < rad**2:
            region[i][j] = True
constraint_cylinder = VelocityConstraint2D((Nx, Ny), (0.0, 0.0), region)

integrator = SSmoke2D((Nx, Ny), (dx, dy), hbar, periodic=True, initValues=initial_vel_field,
                      constraints=(constraint_cylinder,))
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
print('Calculating flow via SSmoke2D...')
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
save_2d('prerendered\\SSmoke2DStationaryCylinder.data', Nframes, (Nx, Ny), (dx, dy), times, xcoords, ycoords, vx, vy)
print('Saving complete.')



