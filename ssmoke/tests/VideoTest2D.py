import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from ..common import *
from ..ssmoke2d import SSmoke2D
from ..dataio import save_2d
from ..visualizer import animation_2d_magcurl

#====================================================================================================
# Checking if:
#   1. dataio module works correctly in saving and loading data
#   2. FFMpeg works correctly with custom visualizer animations
#====================================================================================================

Nframes = 50

def initial_vel_field(pos):
    x, y = pos
    if ((x - 2.5)**2 + (y - 2.5)**2) < 1.5625:
        return (1.0, 0.0)
    elif ((x - 7.5)**2 + (y - 2.5)**2) < 1.5625:
        return (0.0, 0.0)
    return (0.0, 0.0)

integrator = SSmoke2D((200, 100), 0.05, 0.02, True, initValues=initial_vel_field)
xcoords, ycoords = integrator.meshgrid()
vx = [None] * Nframes
vy = [None] * Nframes
curls = [None] * Nframes
times = [None] * Nframes
vx[0], vy[0] = integrator.flow_vel()
times[0] = 0.0

print()
print('Calculating flow via SSmoke2D...')
for frame in range(1, Nframes):
    print_progress_bar(frame, 0, Nframes)
    integrator.advance_timestep(0.02)
    vx[frame], vy[frame] = integrator.flow_vel()
    times[frame] = round_sig(frame * 0.02)
print()
print('Calculation complete.')
print()

print('Saving data...')
save_2d('prerendered\\VideoTest2D.data', Nframes, (200, 100), (0.05, 0.05), times, xcoords, ycoords, vx, vy)
print('Data saved.')
print()

anim = animation_2d_magcurl('prerendered\\VideoTest2D.data')
plt.show()

print('Writing animation...')
anim = animation_2d_magcurl('prerendered\\VideoTest2D.data')
videoWriter = FFMpegFileWriter(fps=10)
anim.animation.save('prerendered\\VideoTest2D.mp4', videoWriter)
print('Writing complete.')
print()


