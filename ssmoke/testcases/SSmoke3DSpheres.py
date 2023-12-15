import matplotlib.pyplot as plt
from ..common import *
from ..ssmoke3d import SSmoke3D
from ..dataio import save_3d
from ..visualizer import image_2d_magcurl_raw

#====================================================================================================
# Initial condition: 'spheres' of fluid moving left and right
#====================================================================================================
# Parameters
#----------------------------------------------------------------------------------------------------

Nx = 200
Ny = 100
Nz = 100
Nframes = 200
steps_per_frame = 10
Nsteps = Nframes * steps_per_frame

dx = 0.01
dy = 0.01
dz = 0.01
dt = 0.005
hbar = 0.005
Lx = Nx * dx
Ly = Ny * dy
Lz = Nz * dz

#----------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------

Cx1 = 0.25 * Lx
Cx2 = 0.75 * Lx
Cy = 0.5 * Ly
Cz = 0.5 * Lz
rad = min(min(0.5 * Cx1, 0.5 * Cy), 0.5 * Cz)

def initial_vel_field(pos):
    x, y, z = pos
    if ((x - Cx1)**2 + (y - Cy)**2 + (z - Cz)**2) < rad**2:
        return (1.0, 0.0, 0.0)
    elif ((x - Cx2)**2 + (y - Cy)**2 + (z - Cz)**2) < rad**2:
        return (-1.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0)

integrator = SSmoke3D((Nx, Ny, Nz), (dx, dy, dz), hbar, periodic=True, initValues=initial_vel_field)
xcoords, ycoords, zcoords = integrator.meshgrid()
vx = [None] * Nframes
vy = [None] * Nframes
vz = [None] * Nframes
times = [None] * Nframes
vx[0], vy[0], vz[0] = integrator.flow_vel()
times[0] = 0.0

#----------------------------------------------------------------------------------------------------
# Simulation
#----------------------------------------------------------------------------------------------------

print()
print('Calculating flow via SSmoke3D...')
for frame in range(1, Nframes):
    for t in range(steps_per_frame):
        print_progress_bar(frame * steps_per_frame + t, 0, Nsteps)
        integrator.advance_timestep(dt)
    vx[frame], vy[frame], vz[frame] = integrator.flow_vel()
    times[frame] = round_sig(frame * steps_per_frame * dt)
print()
print('Calculation complete.')
print()

#----------------------------------------------------------------------------------------------------
# Saving data
#----------------------------------------------------------------------------------------------------

print('Saving data...')
save_3d('prerendered\\SSmoke3DSpheres.data', Nframes, (Nx, Ny, Nz), (dx, dy, dz), times, xcoords, ycoords, zcoords, vx, vy, vz)
print('Saving complete.')

#----------------------------------------------------------------------------------------------------
# (For debugging) static preview
#----------------------------------------------------------------------------------------------------

frames_to_preview = [0, int(Nframes//4), int(Nframes//2), int(3 * Nframes//4), Nframes - 1]
zslice = int(Nz//2)

for f in frames_to_preview:
    fig, ax = image_2d_magcurl_raw(xcoords[:,:,zslice], ycoords[:,:,zslice], vx[f][:,:,zslice], vy[f][:,:,zslice])
    plt.show()
