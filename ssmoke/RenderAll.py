import os
from matplotlib.animation import FFMpegFileWriter
from .visualizer import *

#====================================================================================================
# A script to render all of the .data files in this folder into mp4 files.
#====================================================================================================

print()
print(r'Running RenderAll script...')
def check_necessary(name: str):
    return (os.path.exists(os.path.join('prerendered', name + '.data')) and 
            not os.path.exists(os.path.join('report', name + '.mp4')))

videoWriter = FFMpegFileWriter(fps=10)

#----------------------------------------------------------------------------------------------------

if check_necessary('Stam2DVortices'):
    print(r'    Stam2DVortices detected as needing processing.')
    anim = animation_2d_magcurl('prerendered\\Stam2DVortices.data')
    anim.animation.save('report\\Stam2DVortices.mp4', videoWriter)
    print(r'        Done processing.')
    
#----------------------------------------------------------------------------------------------------

if check_necessary('Stam2DCylinders'):
    print(r'    Stam2DCylinders detected as needing processing.')
    anim = animation_2d_magcurl('prerendered\\Stam2DCylinders.data')
    anim.animation.save('report\\Stam2DCylinders.mp4', videoWriter)
    print(r'        Done processing.')
    
#----------------------------------------------------------------------------------------------------

if check_necessary('SSmoke2DVortices'):
    print(r'    SSmoke2DVortices detected as needing processing.')
    anim = animation_2d_magcurl('prerendered\\SSmoke2DVortices.data')
    anim.animation.save('report\\SSmoke2DVortices.mp4', videoWriter)
    print(r'        Done processing.')
    
#----------------------------------------------------------------------------------------------------

if check_necessary('SSmoke2DCylinders'):
    print(r'    SSmoke2DCylinders detected as needing processing.')
    anim = animation_2d_magcurl('prerendered\\SSmoke2DCylinders.data')
    anim.animation.save('report\\SSmoke2DCylinders.mp4', videoWriter)
    print(r'        Done processing. (1/2)')

    rng = np.random.default_rng()
    initialX_blue = []
    initialY_blue = []
    initialX_red = []
    initialY_red = []
    for _ in range(5000):
        tX = 20.0 * rng.random()
        tY = 10.0 * rng.random()
        if ((tX - 5)**2 + (tY - 5)**2 < 6.25) or ((tX - 15)**2 + (tY - 5)**2 < 6.25):
            initialX_red.append(tX)
            initialY_red.append(tY)
        else:
            initialX_blue.append(tX)
            initialY_blue.append(tY)

    anim = animation_2d_advection('prerendered\\SSmoke2DCylinders.data', (initialX_blue, initialY_blue),
                                  startPosRed=(initialX_red, initialY_red))
    anim.animation.save('report\\SSmoke2DCylindersParticles.mp4', videoWriter)
    print(r'        Done processing. (2/2)')
    
#----------------------------------------------------------------------------------------------------

if check_necessary('SSmoke2DStationaryCylinder'):
    print(r'    SSmoke2DStationaryCylinder detected as needing processing.')
    anim = animation_2d_magcurl('prerendered\\SSmoke2DStationaryCylinder.data')
    fig = plt.gcf()
    circle0 = plt.Circle((0.5, 0.5), 0.125, color='k')
    circle1 = plt.Circle((0.5, 0.5), 0.120, color='r') #Radius slightly shrunk so that the boundary layer can be seen
    fig.axes[0].add_patch(circle0)
    fig.axes[1].add_patch(circle1)
    anim.animation.save('report\\SSmoke2DStationaryCylinder.mp4', videoWriter)
    print(r'        Done processing.')
    
#----------------------------------------------------------------------------------------------------

if check_necessary('SSmoke2DAirfoil'):
    print(r'    SSmoke2DAirfoil detected as needing processing.')

    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    zeta = -0.1 + 0.05j + abs(1.1 + 0.05j) * np.exp(1j * theta)
    z = zeta + (1.0 / zeta)
    x = 0.93969 * np.real(z) + 0.34202 * np.imag(z)
    y = 0.93969 * np.imag(z) - 0.34202 * np.real(z)

    anim = animation_2d_magcurl('prerendered\\SSmoke2DAirfoil.data')
    fig = plt.gcf()
    overlay0 = plt.Polygon(np.vstack((0.9 + (0.2 * x), 0.5 + (0.2 * y))).T, color='k')
    overlay1 = plt.Polygon(np.vstack((0.9 + (0.19 * x), 0.5 + (0.19 * y))).T, color='r')
    fig.axes[0].add_patch(overlay0)
    fig.axes[1].add_patch(overlay1)
    anim.animation.save('report\\SSmoke2DAirfoil.mp4', videoWriter)
    print(r'        Done processing.')
    
#----------------------------------------------------------------------------------------------------

if check_necessary('SSmoke2DAirfoilPeriodic'):
    print(r'    SSmoke2DAirfoilPeriodic detected as needing processing.')

    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    zeta = -0.1 + 0.05j + abs(1.1 + 0.05j) * np.exp(1j * theta)
    z = zeta + (1.0 / zeta)
    x = 0.93969 * np.real(z) + 0.34202 * np.imag(z)
    y = 0.93969 * np.imag(z) - 0.34202 * np.real(z)

    anim = animation_2d_magcurl('prerendered\\SSmoke2DAirfoilPeriodic.data')
    fig = plt.gcf()
    overlay0 = plt.Polygon(np.vstack((0.9 + (0.2 * x), 0.75 + (0.2 * y))).T, color='k')
    overlay1 = plt.Polygon(np.vstack((0.9 + (0.19 * x), 0.75 + (0.19 * y))).T, color='r')
    fig.axes[0].add_patch(overlay0)
    fig.axes[1].add_patch(overlay1)
    anim.animation.save('report\\SSmoke2DAirfoilPeriodic.mp4', videoWriter)
    print(r'        Done processing.')
    
#----------------------------------------------------------------------------------------------------
# Warning: this one takes ~4hrs to process!

if check_necessary('SSmoke3DSpheres'):
    print(r'    SSmoke3DSpheres detected as needing processing.')
    anim = animation_3d_curl('prerendered\\SSmoke3DSpheres.data')
    anim.animation.save('report\\SSmoke3DSpheres.mp4', videoWriter)
    print(r'        Done processing. (1/2)')

    rng = np.random.default_rng()
    initialX_blue = []
    initialY_blue = []
    initialZ_blue = []
    initialX_red = []
    initialY_red = []
    initialZ_red = []
    for _ in range(5000):
        tX = 2.0 * rng.random()
        tY = rng.random()
        tZ = rng.random()
        if ((tX - 0.5)**2 + (tY - 0.5)**2 + (tZ - 0.5)**2 < 0.0625):
            initialX_red.append(tX)
            initialY_red.append(tY)
            initialZ_red.append(tZ)
        elif ((tX - 1.5)**2 + (tY - 0.5)**2 + (tZ - 0.5)**2 < 0.0625):
            initialX_blue.append(tX)
            initialY_blue.append(tY)
            initialZ_blue.append(tZ)

    anim = animation_3d_advection('prerendered\\SSmoke3DSpheres.data', (initialX_blue, initialY_blue, initialZ_blue),
                                  startPosRed=(initialX_red, initialY_red, initialZ_red))
    anim.animation.save('report\\SSmoke3DSpheresParticles.mp4', videoWriter)
    print(r'        Done processing. (2/2)')
    
#----------------------------------------------------------------------------------------------------
# Warning: this one takes ~45mins to process!

if check_necessary('SSmoke3DStationarySphere'):
    print(r'    SSmoke3DStationarySphere detected as needing processing.')
    anim = animation_3d_curl('prerendered\\SSmoke3DStationarySphere.data')

    phi, theta = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    sphere_x = 0.5 + (0.125 * np.cos(phi) * np.sin(theta))
    sphere_y = 0.5 + (0.125 * np.sin(phi) * np.sin(theta))
    sphere_z = 0.5 + (0.125 * np.cos(theta))

    fig = plt.gcf()
    fig.axes[0].plot_surface(sphere_x, sphere_y, sphere_z, color='r')

    anim.animation.save('report\\SSmoke3DStationarySphere.mp4', videoWriter)
    print(r'        Done processing.')
    
#----------------------------------------------------------------------------------------------------

print(r'RenderAll script complete. Exiting...')
print()
print()
