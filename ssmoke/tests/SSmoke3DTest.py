import matplotlib.pyplot as plt
from ..common import *
from ..ssmoke3d import SSmoke3D
from ..dataio import save_3d
from ..visualizer import animation_3d_advection

#====================================================================================================
# Testcase: trivial flow ux = 1 everywhere. Testing to see if SSmoke3D class functions as advertised,
# and does not produce any unexpected errors.
#====================================================================================================

Nx = 100
Ny = 50
Nz = 50
Nframes = 25
steps_per_frame = 5
Nsteps = Nframes * steps_per_frame

dx = 0.1
dy = 0.1
dz = 0.1
dt = 0.02
hbar = 0.02
Lx = Nx * dx
Ly = Ny * dy

def initial_vel_field(pos):
    return (1.0, 0.0, 0.0)

integrator = SSmoke3D((Nx, Ny, Nz), (dx, dy, dz), hbar=hbar, periodic=True, initValues=initial_vel_field)
xcoords, ycoords, zcoords = integrator.meshgrid()
vx = [None] * Nframes
vy = [None] * Nframes
vz = [None] * Nframes
times = [None] * Nframes
vx[0], vy[0], vz[0] = integrator.flow_vel()
times[0] = 0.0

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

print('Saving data...')
save_3d('prerendered\\SSmoke3DTest.data', Nframes, (Nx, Ny, Nz), (dx, dy, dz), times, xcoords, ycoords, zcoords, vx, vy, vz)
print('Saving complete.')

anim = animation_3d_advection('prerendered\\SSmoke3DTest.data')
plt.show()


