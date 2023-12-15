import time
import matplotlib.pyplot as plt
from ..common import *
from ..ssmoke2d import SSmoke2D

#====================================================================================================
# Testcase: trivial flow ux = 1, uy = 0 everywhere. Testing to see if SSmoke2D class functions as
# advertised, and does not produce any unexpected errors.
#====================================================================================================

Nx = 100
Ny = 50

dx = 0.1
dy = 0.1
hbar = 0.02
dt = 0.02

def initial_vel_field(pos):
    return (1.0, 0.0)

integrator = SSmoke2D((Nx, Ny), (dx, dy), hbar, True, initial_vel_field)
xcoords, ycoords = integrator.meshgrid()
vx, vy = integrator.flow_vel()

skip = (slice(None, None, 3), slice(None, None, 3)) # Reduced sample space for pyplot.quiver()
fig, axes = plt.subplots(2, 1)
axes[0].streamplot(xcoords, ycoords, vx.transpose(), vy.transpose(), density=0.5)
axes[0].set_aspect((Ny * dy) / (Nx * dx))
axes[1].quiver(xcoords[skip], ycoords[skip], vx.transpose()[skip], vy.transpose()[skip], angles='xy', scale_units='xy')
axes[1].set_aspect((Ny * dy) / (Nx * dx))
plt.show()

#----------------------------------------------------------------------------------------------------
# Can we perform a timestep without exploding?

integrator.advance_timestep(dt)
vx, vy = integrator.flow_vel()

fig, axes = plt.subplots(2, 1)
axes[0].streamplot(xcoords, ycoords, vx.transpose(), vy.transpose(), density=0.5)
axes[0].set_aspect((Ny * dy) / (Nx * dx))
axes[1].quiver(xcoords[skip], ycoords[skip], vx.transpose()[skip], vy.transpose()[skip], angles='xy', scale_units='xy')
axes[1].set_aspect((Ny * dy) / (Nx * dx))
plt.show()

#----------------------------------------------------------------------------------------------------
# The velocity field should not evolve at all

for _ in range(50):
    integrator.advance_timestep(dt)
vx, vy = integrator.flow_vel()

fig, axes = plt.subplots(2, 1)
axes[0].streamplot(xcoords, ycoords, vx.transpose(), vy.transpose(), density=0.5)
axes[0].set_aspect((Ny * dy) / (Nx * dx))
axes[1].quiver(xcoords[skip], ycoords[skip], vx.transpose()[skip], vy.transpose()[skip], angles='xy', scale_units='xy')
axes[1].set_aspect((Ny * dy) / (Nx * dx))
plt.show()

#----------------------------------------------------------------------------------------------------
# Timing test

n_trials = 100
start = time.time()
for _ in range(n_trials):
    integrator.advance_timestep(dt)
end = time.time()
print()
print(f'Domain size: {Nx} x {Ny}, calculation time: {round_sig(1000 * (end - start) / n_trials)} ms per timestep.')
print()
print()
input(r'Press Enter to quit...')