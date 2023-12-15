from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Any, Callable, Iterable
from numpy.typing import ArrayLike, NDArray
from .common import *

#====================================================================================================
# The Stam2D class represents a single 'timestate' of a IIF flow field, i.e. contains all of the
# necessary information to describe a incompressible and inviscid velocity field over a two-
# dimensional rectangular domain, as well as to propagate it into the next timestep using Stam's
# Stable Fluids method. An instance of the Stam2D class stores the following information:
#
#     - Domain information:
#         - Two integers (Nx, Ny) denoting the number of spatial 'gridcells' used in each dimension.
#         - Two floats (dx, dy) denoting the size of each gridcell in each dimension.
#
#     - Wavefunction information:
#         - Two numpy.ndarrays of real numbers, representing the x and y components of the flow
#             velocity vector at the center of each gridcell.
#
# Notationally, a distinction is made between the 'real space' comprised of continuous coordinates
# (x, y) and the discrete 'lattice' of gridcells specified by integers (i, j); all functions are thus
# named according to this distinction.
#
# The integrator defaults to periodic boundary conditions. Setting the 'periodic' parameter to false
# allows open boundary conditions to be used (interpreted as velocity gradients being 0 at and beyond
# beyond the boundary), but this may not correctly correspond to physically meaningful boundary
# conditions.
#====================================================================================================

class Stam2D:
    
    # Instance initializer; a Stam2D can be initialized using the following parameters:
    #
    #     - lattice_size: By default, this should be a tuple of ints (Nx, Ny). If a single integer N
    #                       is supplied, this will be interpreted as (N, N). This parameter is
    #                       ignored if initValues is specified as a Stam2D object.
    #
    #     - resolution:   By default, this should be a tuple of floats (dx, dy). If a single float dx
    #                       is supplied, this will be interpreted as (dx, dx). This parameter is
    #                       ignored if initValues is specified as a Stam2D object.
    #
    #     - periodic:     Whether the boundary conditions are periodic (true) or open (false).
    #
    #     - initValues:   Optional argument; it supplies the following behaviours:
    #                         - If unspecified, creates a flow field with zero velocity everywhere.
    #                         - If given as a numpy.ndarray of shape (2, Nx, Ny), copies the supplied
    #                             values directly into the flow field.
    #                         - If given as a single-parameter function, which maps a tuple of floats
    #                             (x, y) to a tuple of floats (vx, vy), creates a flow field with
    #                             initial velocity distributed according to the supplied function.
    #                         - If given as a Stam2D object, creates a shallow copy.
    #                     Providing any other type of input may result in exceptions being thrown.

    def __init__(self, lattice_size: int | Iterable[int] = 1,
                 resolution: float | Iterable[float] = 1.0,
                 periodic: bool = True,
                 initValues: ArrayLike | Callable[[Iterable[float]], Any] | Stam2D = None) -> None:
        
        if isinstance(initValues, Stam2D):
            self.shape = initValues.shape                    # Parameter
            self.res = initValues.res                        # Parameter
            self.__multi_iter = initValues.__multi_iter      # Memoization, for internal optimization
            self.__kx = initValues.__kx                      # Memoization, for internal optimization
            self.__ky = initValues.__ky                      # Memoization, for internal optimization
            self.__ksq = initValues.__ksq                    # Memoization, for internal optimization
            self.periodic = initValues.periodic              # Parameter
            self.__fieldX = initValues.__fieldX.copy()       # Parameter
            self.__fieldY = initValues.__fieldY.copy()       # Parameter
        else:
            
            self.shape: tuple[int, int] = coerce_tuple(lattice_size, ndims=2, dtype=int)
            self.res: tuple[float, float] = coerce_tuple(resolution, ndims=2, dtype=float)
            
            self.__multi_iter = [(i,j) for i in range(self.shape[0]) for j in range(self.shape[1])]
            self.__kx = None
            self.__ky = None
            self.__ksq = None
            self.periodic = bool(periodic)

            if initValues is None:
                self.__fieldX = np.zeros(self.shape, dtype=float)
                self.__fieldY = np.zeros(self.shape, dtype=float)

            elif isinstance(initValues, Callable):

                test = initValues((0.0, 0.0))
                self.__fieldX = np.zeros(self.shape, dtype=float)
                self.__fieldY = np.zeros(self.shape, dtype=float)

                if isinstance(test, Iterable) and isscalar(test[0]):
                    for i, j in self.__multi_iter:
                        x = i * self.res[0]
                        y = j * self.res[1]
                        self.__fieldX[i][j], self.__fieldY[i][j] = initValues((x, y))
                else:
                    raise TypeError(r'Input initValues is a function of wrong output type!')

            elif isinstance(initValues, np.ndarray):

                if initValues.shape == (2, self.shape[0], self.shape[1]):
                    self.__fieldX = initValues[0].astype(float)
                    self.__fieldY = initValues[1].astype(float)
                else:
                    raise IndexError(r'Input initValues is a ndarray of wrong shape!')

            elif isinstance(initValues, Iterable) and isinstance(initValues[0], np.ndarray):

                if len(initValues) == 2:
                    self.__fieldX = initValues[0].astype(float)
                    self.__fieldY = initValues[1].astype(float)
                else:
                    raise IndexError(r'Input initValues is an iterable of too few ndarrays!')

            else:
                raise TypeError(r'Input initValues is not recognized as a valid input type!')

    #------------------------------------------------------------------------------------------------
    # Returns two 1D arrays representing the x and y coordinates of every gridcell; i.e. the numpy- 
    # style meshgrid over this domain. The intended notation is x,y = this.meshgrid().

    def meshgrid(self) -> list[NDArray[Any], NDArray[Any]]:
        xcoords = [i * self.res[0] for i in range(self.shape[0])]
        ycoords = [j * self.res[1] for j in range(self.shape[1])]
        return np.meshgrid(xcoords, ycoords, indexing='ij')
    
    #------------------------------------------------------------------------------------------------
    # Returns two 1D arrays representing the vx and vy components of the flow velocity at every
    # gridcell, corresponding to the meshgrid. The intended notation is vx,vy = this.flow_vel().

    def flow_vel(self) -> tuple[NDArray[Any], NDArray[Any]]:
        return (self.__fieldX, self.__fieldY)
        
    #------------------------------------------------------------------------------------------------
    # Advances the flow field by a timestep dt.

    def advance_timestep(self, dt: float) -> None:
        
        vX, vY = self.advect_field(self.__fieldX, self.__fieldY, dt)
        ftX = np.fft.fft2(vX)
        ftY = np.fft.fft2(vY)

        if self.__kx is None or self.__ky is None or self.__ksq is None:

            kxvals = np.fft.fftfreq(self.shape[0], self.res[0]) # This is a 1D array
            kyvals = np.fft.fftfreq(self.shape[1], self.res[1]) # This is a 1D array

            self.__kx = np.empty(self.shape)  # This is a 2D array
            self.__ky = np.empty(self.shape)  # This is a 2D array
            self.__ksq = np.empty(self.shape) # This is a 2D array

            for i, j in self.__multi_iter:
                self.__kx[i][j] = kxvals[i]
                self.__ky[i][j] = kyvals[j]
                self.__ksq[i][j] = (kxvals[i])**2 + (kyvals[j])**2
        
        kdotv = (self.__kx * ftX) + (self.__ky * ftY)
        ftX -= np.divide(self.__kx * kdotv, self.__ksq, out=np.zeros_like(ftX), where=(self.__ksq>0.0))
        ftY -= np.divide(self.__ky * kdotv, self.__ksq, out=np.zeros_like(ftY), where=(self.__ksq>0.0))

        self.__fieldX = np.real(np.fft.ifft2(ftX)) # Need to enforce real output; otherwise imaginary
        self.__fieldY = np.real(np.fft.ifft2(ftY)) # values may creep in by numerical inaccuracy
        
    #------------------------------------------------------------------------------------------------
    # Advects a field (fx, fy) by the flow velocity over a timescale dt. This function returns a
    # tuple (fx', fy') containing the x and y components of the advected field. Note the the input
    # field must have the same shape as the lattice!

    def advect_field(self, fx: NDArray[Any], fy: NDArray[Any], dt: float) -> None:

        if fx.shape != self.shape or fy.shape != self.shape:
            raise TypeError('Input field has the wrong shape!')

        arrayI = np.linspace(0, self.shape[0] - 1, self.shape[0])
        arrayJ = np.linspace(0, self.shape[1] - 1, self.shape[1])
        coordI, coordJ = np.meshgrid(arrayI, arrayJ, indexing='ij')

        backtracedI = coordI - (dt * self.__fieldX / self.res[0])
        backtracedJ = coordJ - (dt * self.__fieldY / self.res[1])

        if self.periodic:

            arrayI = np.linspace(-1, self.shape[0], self.shape[0] + 2)
            arrayJ = np.linspace(-1, self.shape[1], self.shape[1] + 2)
            extendedfx = np.pad(fx, (1, 1), mode='wrap')
            extendedfy = np.pad(fy, (1, 1), mode='wrap')
            interpx = RegularGridInterpolator((arrayI, arrayJ), extendedfx)
            interpy = RegularGridInterpolator((arrayI, arrayJ), extendedfy)
            backtracedI = backtracedI % self.shape[0]
            backtracedJ = backtracedJ % self.shape[1]

        else:

            interpx = RegularGridInterpolator((arrayI, arrayJ), fx)
            interpy = RegularGridInterpolator((arrayI, arrayJ), fy)
            backtracedI = np.clip(backtracedI, 0, self.shape[0] - 1)
            backtracedJ = np.clip(backtracedJ, 0, self.shape[1] - 1)

        return (interpx((backtracedI, backtracedJ)), interpy((backtracedI, backtracedJ)))


