from ..common import *
from ..ssmoke2d import SSmoke2D
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
dt = 0.02
hbar = 0.005
Lx = Nx * dx
Ly = Ny * dy

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

Cx1 = 0.25 * Lx
Cx2 = 0.75 * Lx
Cy = 0.5 * Ly
thickness = 0.1 * Ly

def initial_vel_field(pos):
    x, y = pos
    if abs(y - Cy) < thickness:
        return ((0.0, 1.0) if (x < Cx1 or x > Cx2) else (0.0, -1.0))
    return (0.0, 0.0)

integrator = SSmoke2D((Nx, Ny), (dx, dy), hbar, periodic=True, initValues=initial_vel_field)
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
# Saving data
#----------------------------------------------------------------------------------------------------

print('Saving data...')
save_2d('prerendered\\SSmoke2DVortices.data', Nframes, (Nx, Ny), (dx, dy), times, xcoords, ycoords, vx, vy)
print('Saving complete.')




