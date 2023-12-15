import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from ..common import *
from ..ssmoke2d import SSmoke2D
from ..dataio import save_2d, delete_data
from ..visualizer import animation_2d_magcurl

#====================================================================================================
# Testcase: a simple irrotational vortex at the origin ux = -y/r^2, uy = x/r^2. Note that this vortex
# is centred at the bottom left corner of a periodic 'box', but the vortex itself fails to respect
# periodicity! This vortex should be thus be 'unstable' especially at the boundaries, and evolve in a
# possibly nontrivial manner.
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nx = 200
Ny = 200
Nframes = 100
steps_per_frame = 10
Nsteps = Nframes * steps_per_frame

dx = 0.05
dy = 0.05
dt = 0.025
hbar = 0.025

def curl_field(ux, uy):
    padded_vel = np.pad(ux, (1, 1), 'wrap')
    duxdy = np.gradient(padded_vel, dx, dy)[1][1:-1,1:-1]
    padded_vel = np.pad(uy, (1, 1), 'wrap')
    duydx = np.gradient(padded_vel, dx, dy)[0][1:-1,1:-1]
    return (duydx - duxdy)

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

# Note: the velocity field gets transformed into:
#     ux -> ux + (dux/dx)x + (duy/dx)y
#     uy -> uy + (dux/dy)x + (duy/dy)y
# due to the nonlinear terms not accounted for inside SSmoke2D.
# The (ux, uy) chosen below transforms into (-y/r^2, x/r^2).

def initial_vel_field(pos):
    x = pos[0]
    y = pos[1]
    return ((0.0, -math.atan2(x, y) / y) if y != 0.0 else (0.0, 0.0))

integrator = SSmoke2D((Nx, Ny), (dx, dy), hbar, True, initial_vel_field)
xcoords, ycoords = integrator.meshgrid()
vx = [None] * Nframes
vy = [None] * Nframes
curls = [None] * Nframes
times = [None] * Nframes
vx[0], vy[0] = integrator.flow_vel()
curls[0] = curl_field(vx[0], vy[0])
times[0] = 0

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
    curls[frame] = curl_field(vx[frame], vy[frame])
    times[frame] = round_sig(frame * steps_per_frame * dt)
print()
print('Calculation complete.')
print()

#----------------------------------------------------------------------------------------------------
# Matplotlib animation
#----------------------------------------------------------------------------------------------------

save_2d('prerendered\\tmp_Vortex2D.data', Nframes, (Nx, Ny), (dx, dy), times, xcoords, ycoords, vx, vy)
animation = animation_2d_magcurl('prerendered\\tmp_Vortex2D.data')
plt.show()
delete_data('prerendered\\tmp_Vortex2D.data')


