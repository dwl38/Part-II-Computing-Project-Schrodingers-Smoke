import numpy as np
from ..common import *
from ..ssmoke2d import SSmoke2D, VelocityConstraint2D
from ..dataio import save_2d

#====================================================================================================
# Initial condition: fluid past a Joukowski airfoil, with non-zero angle of attack
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nx = 625
Ny = 375
Nframes = 80
steps_per_frame = 10
Nsteps = Nframes * steps_per_frame

dx = 0.004
dy = 0.004
dt = 0.002
hbar = 0.005
Lx = Nx * dx
Ly = Ny * dy

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

Cx = 0.36 * Lx
Cy = 0.5 * Ly

scale = 0.2                 # Joukowski airfoil canonically extends from x = -2 to +2, need to rescale
aoa = 0.3490659             # rads, equiv to 20 degrees
jkwsk_offset = 0.1 - 0.05j  # Parameter for Joukowski transformation

def joukowski_airfoil(x, y):
    x_scaled = (x - Cx) / scale
    y_scaled = (y - Cy) / scale
    x_rot = (x_scaled * np.cos(aoa)) - (y_scaled * np.sin(aoa))
    y_rot = (x_scaled * np.sin(aoa)) + (y_scaled * np.cos(aoa))
    z = complex(real=(x_rot/2), imag=(y_rot/2))
    return ((abs(z + np.sqrt((z**2) - 1) + jkwsk_offset) < 1 + abs(jkwsk_offset))
            and (abs(z - np.sqrt((z**2) - 1) + jkwsk_offset) < 1 + abs(jkwsk_offset)))

def initial_vel_field(pos):
    x, y = pos
    if joukowski_airfoil(x, y):
        return (0.0, 0.0)
    return (1.0, 0.0)

region = np.zeros((Nx, Ny), dtype=bool)
for i in range(Nx):
    for j in range(Ny):
        region[i][j] = joukowski_airfoil(i * dx, j * dy)
constraint_airfoil = VelocityConstraint2D((Nx, Ny), (0.0, 0.0), region)

integrator = SSmoke2D((Nx, Ny), (dx, dy), hbar, periodic=True, initValues=initial_vel_field,
                      constraints=(constraint_airfoil,))
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
save_2d('prerendered\\SSmoke2DAirfoilPeriodic.data', Nframes, (Nx, Ny), (dx, dy), times, xcoords, ycoords, vx, vy)
print('Saving complete.')



