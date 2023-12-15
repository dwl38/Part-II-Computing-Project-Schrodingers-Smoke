import numpy as np
from ..common import *
from ..ssmoke2d import SSmoke2D, VelocityConstraint2D
from ..dataio import save_2d

#====================================================================================================
# Initial condition: jet of fluid injected into stationary medium
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nx = 400
Ny = 200
Nframes = 200
steps_per_frame = 10
Nsteps = Nframes * steps_per_frame

dx = 0.05
dy = 0.05
dt = 0.02
hbar = 0.02
Lx = Nx * dx
Ly = Ny * dy

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

W = 0.1 * Lx
Cy = 0.5 * Ly

def initial_vel_field(pos):
    x, y = pos
    if x < W and abs(y - Cy) < (W / 2):
        return (1.0, 0.0)
    return (0.0, 0.0)

obstacle = np.zeros((Nx, Ny), dtype=bool)
for i in range(Nx):
    for j in range(Ny):
        if (i * dx) < W and abs((j * dy) - Cy) < (W / 2):
            obstacle[i][j] = True
constraint = VelocityConstraint2D((Nx, Ny), (1.0, 0.0), obstacle)

integrator = SSmoke2D((Nx, Ny), (dx, dy), hbar, periodic=True, initValues=initial_vel_field, constraints=(constraint,))
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
save_2d('prerendered\\SSmoke2DJet.data', Nframes, (Nx, Ny), (dx, dy), times, xcoords, ycoords, vx, vy)
print('Saving complete.')



