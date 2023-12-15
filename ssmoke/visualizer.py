import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator
from typing import Any, Iterable
from numpy.typing import NDArray
from .common import *
from .dataio import load_data

#====================================================================================================
# An interface for producing visual output via Matplotlib/Pyplot; largely relies on the custom
# PausableAnimation class, which is a FuncAnimation with the additional functionality of pausing upon
# user click.
#
# The provided functions only create and load a relevant graphics object (either a (fig, ax) tuple
# for static images, or a PausableAnimation object for animations), and then return it as output; the
# intended usage is to save the function's output to memory, and then call Pyplot's relevant display
# routines, e.g.:
#
#         > anim = visualizer.animation_2d_magcurl('source.data')
#         > pyplot.show()
# 
# The images/animations may be modified before display by directly accessing the figure via Pyplot.
#====================================================================================================


#----------------------------------------------------------------------------------------------------
# The PausableAnimation class merely serves as a wrapper for Matplotlib's FuncAnimation, but with the
# additional capability of pausing the animation upon user click. Used as the main type of animation
# produced by this visualizer module.

class PausableAnimation:
    
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None, save_count=None, *,
                 cache_frame_data=True, **kwargs) -> None:

        self.animation = FuncAnimation(fig, func, frames, init_func, fargs, save_count,
                                       cache_frame_data=cache_frame_data, **kwargs)
        self.paused = False
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs) -> None:
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

#----------------------------------------------------------------------------------------------------
# Generates a PausableAnimation from a data file, using the 'magcurl' plot style; see
# image_2d_magcurl_raw() for a description of the plot style.

def animation_2d_magcurl(filename: str, *, interval: int = 100) -> PausableAnimation:
    
    data = load_data(filename)
    if data['Ndim'] != 2:
        raise TypeError('animation_2d is only intended for 2 dimensional data!')
    
    Nframes = int(data['Nframes'])
    Nx, Ny = data['shape']
    dx, dy = data['res']
    aspect_ratio = (Ny * dy) / (Nx * dx)
    extent = (-0.5 * dx, (Nx - 0.5) * dx, -0.5 * dy, (Ny - 0.5) * dy)

    # A reduced sampling space is used for pyplot.quiver so that ~1000 arrows are displayed on screen
    spacing = max(1, round(np.sqrt(Nx * Ny / 1000)))
    skip = (slice(None, None, spacing), slice(None, None, spacing))

    cx = data['cx'][skip]     # location coordinates (reduced sampling space)
    cy = data['cy'][skip]     # location coordinates (reduced sampling space)
    vx = data['vx']           # flow velocity vector (full space)
    vy = data['vy']           # flow velocity vector (full space)
    vmag = [None] * Nframes   # magnitude of flow velocity (full space)
    ux = [None] * Nframes     # direction of flow velocity (reduced sampling space)
    uy = [None] * Nframes     # direction of flow velocity (reduced sampling space)
    curls = [None] * Nframes  # scalar curl of flow velocity (full space)

    for f in range(Nframes):
        vmag[f] = np.sqrt(np.square(vx[f]) + np.square(vy[f]))
        ux[f] = np.divide(vx[f][skip], vmag[f][skip], out=np.zeros_like(vx[f][skip]), where=(vmag[f][skip] != 0.0))
        uy[f] = np.divide(vy[f][skip], vmag[f][skip], out=np.zeros_like(vy[f][skip]), where=(vmag[f][skip] != 0.0))
        padded_vel = np.pad(vx[f], (1, 1), 'wrap')
        dvxdy = np.gradient(padded_vel, dx, dy)[1][1:-1,1:-1]
        padded_vel = np.pad(vy[f], (1, 1), 'wrap')
        dvydx = np.gradient(padded_vel, dx, dy)[0][1:-1,1:-1]
        curls[f] = dvydx - dvxdy

    if aspect_ratio < 1.0:
        fig, axes = plt.subplots(2, 1)
    else:
        fig, axes = plt.subplots(1, 2)

    axes[0].set_aspect(aspect_ratio)
    axes[1].set_aspect(aspect_ratio)

    magplot = axes[0].imshow(vmag[0].transpose(), origin='lower', extent=extent)
    magplotbar = fig.colorbar(magplot, ax=axes[0])
    arrows = axes[0].quiver(cx, cy, ux[0], uy[0], angles='xy', pivot='middle', scale=50, scale_units='width',
                            units='width', width=0.005, headwidth=100, headaxislength=200, headlength=200)

    curlplot = axes[1].imshow(curls[0].transpose(), cmap='Greys', origin='lower', extent=extent)
    curlplotbar = fig.colorbar(curlplot, ax=axes[1])

    axes[0].set_title('Flow field at t = {}'.format(data['times'][0]))
    axes[1].set_title('Curl field at t = {}'.format(data['times'][0]))
    axes[0].set_axis_off()
    axes[1].set_axis_off()

    def animation_func(frame, *fargs):
        magplot.set_data(vmag[frame].transpose())
        magplotbar.update_normal(magplot)
        arrows.set_UVC(ux[frame], uy[frame])
        curlplot.set_data(curls[frame].transpose())
        curlplotbar.update_normal(curlplot)
        axes[0].set_title('Flow field at t = {}'.format(data['times'][frame]))
        axes[1].set_title('Curl field at t = {}'.format(data['times'][frame]))

    return PausableAnimation(fig, animation_func, frames=Nframes, interval=interval, repeat=True, save_count=Nframes)

#----------------------------------------------------------------------------------------------------
# Displays a 'magcurl' image for a 2D flow field, given raw input parameters:
# 
#     - xcoords, an nparray of shape (Nx, Ny)
#     - ycoords, an nparray of shape (Nx, Ny)
#     - vx, an nparray of shape (Nx, Ny)
#     - vy, an nparray of shape (Nx, Ny)
# 
# which draws two plots side-by-side, one of which shows the direction and magnitude of the flow
# velocity as an arrow plot overlaying an intensity plot, and the other of which shows the magnitude
# of the two-dimensional curl as a grayscale intensity plot.

def image_2d_magcurl_raw(xcoords: NDArray[Any], ycoords: NDArray[Any], vx: NDArray[Any], vy: NDArray[Any]):
    
    if vx.shape != vy.shape or xcoords.shape != ycoords.shape:
        raise TypeError('Inputs must have matching shape!')

    Nx, Ny = vx.shape
    dx = xcoords[1][0] - xcoords[0][0]
    dy = ycoords[0][1] - ycoords[0][0]
    aspect_ratio = (Ny * dy) / (Nx * dx)
    extent = (-0.5 * dx, (Nx - 0.5) * dx, -0.5 * dy, (Ny - 0.5) * dy)

    # A reduced sampling space is used for pyplot.quiver so that ~1000 arrows are displayed on screen
    spacing = max(1, round(np.sqrt(Nx * Ny / 1000)))
    skip = (slice(None, None, spacing), slice(None, None, spacing))

    vmag = np.sqrt(np.square(vx) + np.square(vy))
    ux = np.divide(vx[skip], vmag[skip], out=np.zeros_like(vx[skip]), where=(vmag[skip] != 0.0))
    uy = np.divide(vy[skip], vmag[skip], out=np.zeros_like(vy[skip]), where=(vmag[skip] != 0.0))

    padded_vel = np.pad(vx, (1, 1), 'wrap')
    dvxdy = np.gradient(padded_vel, dx, dy)[1][1:-1,1:-1]
    padded_vel = np.pad(vy, (1, 1), 'wrap')
    dvydx = np.gradient(padded_vel, dx, dy)[0][1:-1,1:-1]
    curls = dvydx - dvxdy

    if aspect_ratio < 1.0:
        fig, axes = plt.subplots(2, 1)
    else:
        fig, axes = plt.subplots(1, 2)

    axes[0].set_aspect(aspect_ratio)
    axes[1].set_aspect(aspect_ratio)

    magplot = axes[0].imshow(vmag.transpose(), origin='lower', extent=extent)
    fig.colorbar(magplot, ax=axes[0])
    axes[0].quiver(xcoords[skip], ycoords[skip], ux, uy, angles='xy', pivot='middle', scale=50, scale_units='width',
                   units='width', width=0.005, headwidth=100, headaxislength=200, headlength=200)

    curlplot = axes[1].imshow(curls.transpose(), cmap='Greys', origin='lower', extent=extent)
    fig.colorbar(curlplot, ax=axes[1])

    axes[0].set_title('Flow field')
    axes[1].set_title('Curl field')
    axes[0].set_axis_off()
    axes[1].set_axis_off()

    return (fig, axes)

#----------------------------------------------------------------------------------------------------
# Displays a 'magcurl' image for a 2D flow field, given as a data file and a specific frame to pick
# out; see image_2d_magcurl_raw().

def image_2d_magcurl(filename: str, frame_no: int):
    
    data = load_data(filename)
    if data['Ndim'] != 2:
        raise TypeError('image_2d is only intended for 2 dimensional data!')

    cx = data['cx']    # location coordinates
    cy = data['cy']    # location coordinates
    vx = data['vx']    # flow velocity vector
    vy = data['vy']    # flow velocity vector
    fn = int(frame_no) # frame number
    
    return image_2d_magcurl_raw(cx, cy, vx[fn], vy[fn])

#----------------------------------------------------------------------------------------------------
# Generates a PausableAnimation from a data file, displaying the motion of light test particles
# advected along the flow. The input parameter, startPos, should be specified as:
# 
#     - A length 2 tuple of iterables (x[i], y[i])
#     - A length > 2 iterable of length 2 tuples (x, y)[i]
#
# denoting the initial positions of the test particles at t = 0. If startPos is not specified, 1000
# test particles will be distributed evenly across the domain instead. Automatically treats the space
# as being periodic (so that particles will not be destroyed upon exiting the boundaries), regardless
# of the actual boundary conditions.
#
# As an optional keyword argument, if startPosRed is provided (with the same format as startPos), an
# additional population of red test particles will be advected alongside the blue particles.

def animation_2d_advection(filename: str, startPos: Iterable[Iterable[Any]] = None, *,
                           startPosRed: Iterable[Iterable[Any]] = None, interval: int = 100) -> PausableAnimation:

    data = load_data(filename)
    if data['Ndim'] != 2:
        raise TypeError('animation_2d is only intended for 2 dimensional data!')
    
    Nframes = int(data['Nframes'])
    Nx, Ny = data['shape']
    dx, dy = data['res']
    Lx = Nx * dx
    Ly = Ny * dy
    aspect_ratio = (Ny * dy) / (Nx * dx)

    vx = data['vx']           # flow velocity vector
    vy = data['vy']           # flow velocity vector
    px = [None] * Nframes     # test particles x-coordinates
    py = [None] * Nframes     # test particles y-coordinates

    if startPos is None:

        # Choosing a grid to start from; aims for ~1000 total particles
        Ntest_x = round(np.sqrt(1000 / aspect_ratio))
        Ntest_y = round(aspect_ratio * Ntest_x)

        px[0] = np.empty(Ntest_x * Ntest_y, dtype=float)
        py[0] = np.empty(Ntest_x * Ntest_y, dtype=float)

        for i in range(Ntest_x):
            for j in range(Ntest_y):
                px[0][(i * Ntest_y) + j] = (i + 0.5) * (Lx / Ntest_x)
                py[0][(i * Ntest_y) + j] = (j + 0.5) * (Ly / Ntest_y)

    elif isinstance(startPos, Iterable) and len(startPos) == 2:

        px[0] = np.array(startPos[0], dtype=float)
        py[0] = np.array(startPos[1], dtype=float)

        if px[0].shape != py[0].shape:
            raise TypeError('Tuple of iterables (x[i], y[i]) supplied but shapes of x and y do not match!')
        if len(px[0].shape) != 1:
            raise TypeError('Tuple of iterables (x[i], y[i]) supplied but x is not one-dimensional!')
        
    elif isinstance(startPos, Iterable) and len(startPos[0]) == 2:

        folded = np.array(startPos, dtype=float)
        px[0] = folded[:,0]
        py[0] = folded[:,1]
        
        if px[0].shape != py[0].shape:
            raise TypeError('Iterable of tuples (x, y)[i] supplied but shapes of x and y do not match!')
        if len(px[0].shape) != 1:
            raise TypeError('Iterable of tuples (x, y)[i] supplied but x is not one-dimensional!')

    if startPosRed is None:
        has_red = False
    elif isinstance(startPosRed, Iterable) and len(startPosRed) == 2:

        has_red = True
        px_red = [None] * Nframes
        py_red = [None] * Nframes
        px_red[0] = np.array(startPosRed[0], dtype=float)
        py_red[0] = np.array(startPosRed[1], dtype=float)

        if px_red[0].shape != py_red[0].shape:
            raise TypeError('Tuple of iterables (x[i], y[i]) supplied but shapes of x and y do not match!')
        if len(px_red[0].shape) != 1:
            raise TypeError('Tuple of iterables (x[i], y[i]) supplied but x is not one-dimensional!')
        
    elif isinstance(startPosRed, Iterable) and len(startPosRed[0]) == 2:

        has_red = True
        folded = np.array(startPosRed, dtype=float)
        px_red = [None] * Nframes
        py_red = [None] * Nframes
        px_red[0] = folded[:,0]
        py_red[0] = folded[:,1]
        
        if px_red[0].shape != py_red[0].shape:
            raise TypeError('Iterable of tuples (x, y)[i] supplied but shapes of x and y do not match!')
        if len(px_red[0].shape) != 1:
            raise TypeError('Iterable of tuples (x, y)[i] supplied but x is not one-dimensional!')

    arrayI = np.linspace(-1, Nx, Nx + 2)
    arrayJ = np.linspace(-1, Ny, Ny + 2)

    for f in range(Nframes - 1):
        dt = data['times'][f + 1] - data['times'][f]
        startI = px[f] / dx
        startJ = py[f] / dy
        interpx = RegularGridInterpolator((arrayI, arrayJ), np.pad(vx[f], (1, 1), mode='wrap'))
        interpy = RegularGridInterpolator((arrayI, arrayJ), np.pad(vy[f], (1, 1), mode='wrap'))
        px[f + 1] = (px[f] + (interpx((startI, startJ)) * dt)) % Lx
        py[f + 1] = (py[f] + (interpy((startI, startJ)) * dt)) % Ly

        if has_red:
            startI = px_red[f] / dx
            startJ = py_red[f] / dy
            px_red[f + 1] = (px_red[f] + (interpx((startI, startJ)) * dt)) % Lx
            py_red[f + 1] = (py_red[f] + (interpy((startI, startJ)) * dt)) % Ly

    fig, axis = plt.subplots()
    line, = axis.plot(px[0], py[0], 'b.')
    if has_red:
        line_red, = axis.plot(px_red[0], py_red[0], 'r.')

    axis.set_xlim((0, Lx))
    axis.set_ylim((0, Ly))
    axis.set_aspect(1.0)
    axis.set_title('Test particles motion at t = {}'.format(data['times'][0]))
    #axis.set_axis_off()

    def animation_func(frame, *fargs):
        line.set_data(px[frame], py[frame])
        if has_red:
            line_red.set_data(px_red[frame], py_red[frame])
        axis.set_title('Test particles motion at t = {}'.format(data['times'][frame]))

    return PausableAnimation(fig, animation_func, frames=Nframes, interval=interval, repeat=True, save_count=Nframes)

#----------------------------------------------------------------------------------------------------
# Generates a PausableAnimation from a data file, using the 'curl' plot style, which is a vector
# quiver plot showing instantaneous vorticity everywhere. This roughly corresponds to a 'smoke-like'
# appearance, since realistic smoke particles would be trapped inside vortex lines, with particle
# density scaling monotonically with vorticity strength.

def animation_3d_curl(filename: str, *, interval: int = 100) -> PausableAnimation:
    
    data = load_data(filename)
    if data['Ndim'] != 3:
        raise TypeError('animation_3d is only intended for 3 dimensional data!')
    
    Nframes = int(data['Nframes'])
    Nx, Ny, Nz = data['shape']
    dx, dy, dz = data['res']
    aspect_ratio_y = (Ny * dy) / (Nx * dx)
    aspect_ratio_z = (Nz * dz) / (Nx * dx)

    cx = data['cx']          # location coordinates
    cy = data['cy']          # location coordinates
    cz = data['cz']          # location coordinates
    vx = data['vx']          # flow velocity vector
    vy = data['vy']          # flow velocity vector
    vz = data['vz']          # flow velocity vector
    curlx = [None] * Nframes # curl vector
    curly = [None] * Nframes # curl vector
    curlz = [None] * Nframes # curl vector
    maxcurlmag = 0           # largest magnitude of curl observed

    for f in range(Nframes):
        grads = np.gradient(np.pad(vx[f], (1, 1), 'wrap'), dx, dy, dz)
        dvxdy = grads[1][1:-1,1:-1,1:-1]
        dvxdz = grads[2][1:-1,1:-1,1:-1]
        grads = np.gradient(np.pad(vy[f], (1, 1), 'wrap'), dx, dy, dz)
        dvydx = grads[0][1:-1,1:-1,1:-1]
        dvydz = grads[2][1:-1,1:-1,1:-1]
        grads = np.gradient(np.pad(vz[f], (1, 1), 'wrap'), dx, dy, dz)
        dvzdx = grads[0][1:-1,1:-1,1:-1]
        dvzdy = grads[1][1:-1,1:-1,1:-1]
        curlx[f] = dvzdy - dvydz
        curly[f] = dvxdz - dvzdx
        curlz[f] = dvydx - dvxdy
        maxcurlmag = max(maxcurlmag, np.amax(np.sqrt(curlx[f]**2 + curly[f]**2 + curlz[f]**2)))

    fig = plt.figure()
    axis = fig.add_subplot(projection='3d')

    scale = 0.5 * min(min(dx, dy), dz) / maxcurlmag
    arrows = axis.quiver(cx, cy, cz, curlx[0], curly[0], curlz[0], length=2*scale, arrow_length_ratio=0.0, pivot='middle')
    axis.set_xlim((0, Nx * dx))
    axis.set_ylim((0, Ny * dy))
    axis.set_zlim((0, Nz * dz))
    axis.set_box_aspect((1.0, aspect_ratio_y, aspect_ratio_z))

    def animation_func(frame, *fargs):
        arrow_starts = np.array(((cx - scale * curlx[frame]).flatten(), (cy - scale * curly[frame]).flatten(), (cz - scale * curlz[frame]).flatten()))
        arrow_ends = np.array(((cx + scale * curlx[frame]).flatten(), (cy + scale * curly[frame]).flatten(), (cz + scale * curlz[frame]).flatten()))
        segments = np.array((arrow_starts, arrow_ends)).transpose((2,0,1))
        arrows.set_segments(segments)
        axis.set_title('Vorticity at t = {}'.format(data['times'][frame]))

    animation_func(0)
    return PausableAnimation(fig, animation_func, frames=Nframes, interval=interval, repeat=True, save_count=Nframes)

#----------------------------------------------------------------------------------------------------
# Generates a PausableAnimation from a data file, displaying the motion of light test particles
# advected along the flow. The input parameter, startPos, should be specified as:
# 
#     - A length 3 tuple of iterables (x[i], y[i], z[i])
#     - A length > 3 iterable of length 3 tuples (x, y, z)[i]
#
# denoting the initial positions of the test particles at t = 0. If startPos is not specified, 1000
# test particles will be distributed evenly across the domain instead. Automatically treats the space
# as being periodic (so that particles will not be destroyed upon exiting the boundaries), regardless
# of the actual boundary conditions.
#
# As an optional keyword argument, if startPosRed is provided (with the same format as startPos), an
# additional population of red test particles will be advected alongside the blue particles.

def animation_3d_advection(filename: str, startPos: Iterable[Iterable[Iterable[Any]]] = None, *,
                           startPosRed: Iterable[Iterable[Iterable[Any]]] = None, interval: int = 100) -> PausableAnimation:

    data = load_data(filename)
    if data['Ndim'] != 3:
        raise TypeError('animation_3d is only intended for 3 dimensional data!')
    
    Nframes = int(data['Nframes'])
    Nx, Ny, Nz = data['shape']
    dx, dy, dz = data['res']
    Lx = Nx * dx
    Ly = Ny * dy
    Lz = Nz * dz
    aspect_ratio_y = Ly / Lx
    aspect_ratio_z = Lz / Lx

    vx = data['vx']           # flow velocity vector
    vy = data['vy']           # flow velocity vector
    vz = data['vz']           # flow velocity vector
    px = [None] * Nframes     # test particles x-coordinates
    py = [None] * Nframes     # test particles y-coordinates
    pz = [None] * Nframes     # test particles z-coordinates

    if startPos is None:

        # Choosing a grid to start from; aims for ~1000 total particles
        Ntest_x = round(np.power(1000 / (aspect_ratio_y * aspect_ratio_z), 1/3))
        Ntest_y = round(aspect_ratio_y * Ntest_x)
        Ntest_z = round(aspect_ratio_z * Ntest_x)

        px[0] = np.empty(Ntest_x * Ntest_y * Ntest_z, dtype=float)
        py[0] = np.empty(Ntest_x * Ntest_y * Ntest_z, dtype=float)
        pz[0] = np.empty(Ntest_x * Ntest_y * Ntest_z, dtype=float)

        for i in range(Ntest_x):
            for j in range(Ntest_y):
                for k in range(Ntest_z):
                    unfold = (i * Ntest_y * Ntest_z) + (j * Ntest_z) + k
                    px[0][unfold] = (i + 0.5) * (Lx / Ntest_x)
                    py[0][unfold] = (j + 0.5) * (Ly / Ntest_y)
                    pz[0][unfold] = (k + 0.5) * (Lz / Ntest_z)

    elif isinstance(startPos, Iterable) and len(startPos) == 3:

        px[0] = np.array(startPos[0], dtype=float)
        py[0] = np.array(startPos[1], dtype=float)
        pz[0] = np.array(startPos[2], dtype=float)

        if px[0].shape != py[0].shape or px[0].shape != pz[0].shape:
            raise TypeError('Tuple of iterables (x[i], y[i], z[i]) supplied but shapes do not match!')
        if len(px[0].shape) != 1:
            raise TypeError('Tuple of iterables (x[i], y[i], z[i]) supplied but x is not one-dimensional!')
        
    elif isinstance(startPos, Iterable) and len(startPos[0]) == 3:

        folded = np.array(startPos, dtype=float)
        px[0] = folded[:,0]
        py[0] = folded[:,1]
        pz[0] = folded[:,2]
        
        if px[0].shape != py[0].shape or px[0].shape != pz[0].shape:
            raise TypeError('Iterable of tuples (x, y, z)[i] supplied but shapes do not match!')
        if len(px[0].shape) != 1:
            raise TypeError('Iterable of tuples (x, y, z)[i] supplied but x is not one-dimensional!')

    if startPosRed is None:
        has_red = False
    elif isinstance(startPosRed, Iterable) and len(startPosRed) == 3:

        has_red = True
        px_red = [None] * Nframes
        py_red = [None] * Nframes
        pz_red = [None] * Nframes
        px_red[0] = np.array(startPosRed[0], dtype=float)
        py_red[0] = np.array(startPosRed[1], dtype=float)
        pz_red[0] = np.array(startPosRed[2], dtype=float)

        if px_red[0].shape != py_red[0].shape or px_red[0].shape != pz_red[0].shape:
            raise TypeError('Tuple of iterables (x[i], y[i], z[i]) supplied but shapes do not match!')
        if len(px_red[0].shape) != 1:
            raise TypeError('Tuple of iterables (x[i], y[i], z[i]) supplied but x is not one-dimensional!')
        
    elif isinstance(startPosRed, Iterable) and len(startPosRed[0]) == 3:

        has_red = True
        folded = np.array(startPosRed, dtype=float)
        px_red = [None] * Nframes
        py_red = [None] * Nframes
        pz_red = [None] * Nframes
        px_red[0] = folded[:,0]
        py_red[0] = folded[:,1]
        pz_red[0] = folded[:,2]
        
        if px_red[0].shape != py_red[0].shape or px_red[0].shape != pz_red[0].shape:
            raise TypeError('Iterable of tuples (x, y, z)[i] supplied but shapes do not match!')
        if len(px_red[0].shape) != 1:
            raise TypeError('Iterable of tuples (x, y, z)[i] supplied but x is not one-dimensional!')

    arrayI = np.linspace(-1, Nx, Nx + 2)
    arrayJ = np.linspace(-1, Ny, Ny + 2)
    arrayK = np.linspace(-1, Nz, Nz + 2)

    for f in range(Nframes - 1):
        dt = data['times'][f + 1] - data['times'][f]
        startI = px[f] / dx
        startJ = py[f] / dy
        startK = pz[f] / dz
        interpx = RegularGridInterpolator((arrayI, arrayJ, arrayK), np.pad(vx[f], (1, 1), mode='wrap'))
        interpy = RegularGridInterpolator((arrayI, arrayJ, arrayK), np.pad(vy[f], (1, 1), mode='wrap'))
        interpz = RegularGridInterpolator((arrayI, arrayJ, arrayK), np.pad(vz[f], (1, 1), mode='wrap'))
        px[f + 1] = (px[f] + (interpx((startI, startJ, startK)) * dt)) % Lx
        py[f + 1] = (py[f] + (interpy((startI, startJ, startK)) * dt)) % Ly
        pz[f + 1] = (pz[f] + (interpz((startI, startJ, startK)) * dt)) % Lz

        if has_red:
            startI = px_red[f] / dx
            startJ = py_red[f] / dy
            startK = pz_red[f] / dz
            px_red[f + 1] = (px_red[f] + (interpx((startI, startJ, startK)) * dt)) % Lx
            py_red[f + 1] = (py_red[f] + (interpy((startI, startJ, startK)) * dt)) % Ly
            pz_red[f + 1] = (pz_red[f] + (interpz((startI, startJ, startK)) * dt)) % Lz

    fig = plt.figure()
    axis = fig.add_subplot(projection='3d')
    scatter, = axis.plot(px[0], py[0], pz[0], 'b.')
    if has_red:
        scatter_red, = axis.plot(px_red[0], py_red[0], pz_red[0], 'r.')

    axis.set_xlim((0, Lx))
    axis.set_ylim((0, Ly))
    axis.set_zlim((0, Lz))
    axis.set_box_aspect((1.0, aspect_ratio_y, aspect_ratio_z))
    axis.set_title('Test particles motion at t = {}'.format(data['times'][0]))

    def animation_func(frame, *fargs):
        scatter.set_xdata(px[frame])
        scatter.set_ydata(py[frame])
        scatter.set_3d_properties(pz[frame])
        if has_red:
            scatter_red._offsets3d = (px_red[frame], py_red[frame], pz_red[frame])
            scatter_red.set_xdata(px_red[frame])
            scatter_red.set_ydata(py_red[frame])
            scatter_red.set_3d_properties(pz_red[frame])
        axis.set_title('Test particles motion at t = {}'.format(data['times'][frame]))

    return PausableAnimation(fig, animation_func, frames=Nframes, interval=interval, repeat=True, save_count=Nframes)