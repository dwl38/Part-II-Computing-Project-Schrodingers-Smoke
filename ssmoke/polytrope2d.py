from __future__ import annotations
import numpy as np
from typing import Any, Callable, Iterable
from numpy.typing import ArrayLike, NDArray
from .common import *

#====================================================================================================
# The Polytrope2D class represents a single 'timestate' of a polytropic flow field, i.e. contains all
# of the necessary information to describe a weakly-compressible and inviscid velocity field over a
# two-dimensional rectangular domain. An instance of the Polytrope2D class stores the following
# information:
#
#     - Domain information:
#         - Two integers (Nx, Ny) denoting the number of spatial 'gridcells' used in each dimension.
#         - Two floats (dx, dy) denoting the size of each gridcell in each dimension.
#
#     - Field information:
#         - Two numpy.ndarrays of real numbers, representing the x and y components of the flow
#             velocity vector at the center of each gridcell.
#         - One numpy.ndarray of real numbers, representing the absolute pressure at the centre of
#             each gridcell, where the 'background' value is taken to be 1.
#
# Notationally, a distinction is made between the 'real space' comprised of continuous coordinates
# (x, y) and the discrete 'lattice' of gridcells specified by integers (i, j); all functions are thus
# named according to this distinction.
#
# The integrator defaults to periodic boundary conditions. Setting the 'periodic' parameter to false
# allows open boundary conditions to be used (interpreted as pressure being 1 and velocity gradients
# being 0 at and beyond the boundary), but this may not correctly correspond to physically meaningful
# boundary conditions.
#====================================================================================================

class Polytrope2D:
    
    # Instance initializer; a Polytrope2D can be initialized using the following parameters:
    #
    #     - lattice_size: By default, this should be a tuple of ints (Nx, Ny). If a single integer N
    #                       is supplied, this will be interpreted as (N, N). This parameter is
    #                       ignored if initValues is specified as a Polytrope2D object.
    #
    #     - resolution:   By default, this should be a tuple of floats (dx, dy). If a single float dx
    #                       is supplied, this will be interpreted as (dx, dx). This parameter is
    #                       ignored if initValues is specified as a Polytrope2D object.
    #
    #     - alpha:        A float alpha giving the compressibility of the fluid.
    #
    #     - periodic:     Whether the boundary conditions are periodic (true) or open (false).
    #
    #     - initValues:   Optional argument; it supplies the following behaviours:
    #                         - If unspecified, creates a flow field with zero velocity everywhere.
    #                         - If given as a numpy.ndarray of shape (2, Nx, Ny) or (3, Nx, Ny),
    #                             copies the supplied values directly into the flow field.
    #                         - If given as a single-parameter function, which maps a tuple of floats
    #                             (x, y) to a tuple of floats (vx, vy), creates a flow field with
    #                             initial velocity distributed according to the supplied function.
    #                         - If given as a Polytrope2D object, creates a shallow copy.
    #                     Providing any other type of input may result in exceptions being thrown.

    def __init__(self, lattice_size: int | Iterable[int] = 1,
                 resolution: float | Iterable[float] = 1.0,
                 alpha: float = 0.02,
                 periodic: bool = True,
                 initValues: ArrayLike | Callable[[Iterable[float]], Any] | Polytrope2D = None) -> None:
        
        if isinstance(initValues, Polytrope2D):
            self.shape = initValues.shape                    # Parameter
            self.res = initValues.res                        # Parameter
            self.__multi_iter = initValues.__multi_iter      # Memoization, for internal optimization
            self.alpha = initValues.alpha                    # Parameter
            self.periodic = initValues.periodic              # Parameter
            self.__fieldX = initValues.__fieldX.copy()       # Parameter
            self.__fieldY = initValues.__fieldY.copy()       # Parameter
            self.__fieldP = initValues.__fieldP.copy()       # Parameter
        else:
            
            self.shape: tuple[int, int] = coerce_tuple(lattice_size, ndims=2, dtype=int)
            self.res: tuple[float, float] = coerce_tuple(resolution, ndims=2, dtype=float)
            
            self.__multi_iter = [(i,j) for i in range(self.shape[0]) for j in range(self.shape[1])]
            self.alpha = float(alpha)
            self.periodic = bool(periodic)

            if initValues is None:
                self.__fieldX = np.zeros(self.shape, dtype=float)
                self.__fieldY = np.zeros(self.shape, dtype=float)
                self.__fieldP = np.ones(self.shape, dtype=float)

            elif isinstance(initValues, Callable):

                test = initValues((0.0, 0.0))
                self.__fieldX = np.zeros(self.shape, dtype=float)
                self.__fieldY = np.zeros(self.shape, dtype=float)
                self.__fieldP = np.ones(self.shape, dtype=float)

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
                    self.__fieldP = np.ones(self.shape, dtype=float)
                elif initValues.shape == (3, self.shape[0], self.shape[1]):
                    self.__fieldX = initValues[0].astype(float)
                    self.__fieldY = initValues[1].astype(float)
                    self.__fieldP = initValues[2].astype(float)
                else:
                    raise IndexError(r'Input initValues is a ndarray of wrong shape!')

            elif isinstance(initValues, Iterable) and isinstance(initValues[0], np.ndarray):

                if len(initValues) == 2:
                    self.__fieldX = initValues[0].astype(float)
                    self.__fieldY = initValues[1].astype(float)
                    self.__fieldP = np.ones(self.shape, dtype=float)
                elif len(initValues) > 2:
                    self.__fieldX = initValues[0].astype(float)
                    self.__fieldY = initValues[1].astype(float)
                    self.__fieldP = initValues[2].astype(float)
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
    # Advances the flow field by a timestep dt, either using Euler integration (default; can also be
    # specified by mode strings 'e' or 'euler') or 4th-order Runge-Kutta integration (specified by
    # mode strings 'rk', 'runge', 'kutta', or 'runge-kutta') of the Euler equations.

    def advance_timestep(self, dt: float, mode: str = 'e') -> None:

        if mode.casefold() in {'e', 'euler'}:

            x_k, y_k, p_k = self.calculate_derivative(self.__fieldX, self.__fieldY, self.__fieldP)
            self.__fieldX += x_k * dt
            self.__fieldY += y_k * dt
            self.__fieldP += p_k * dt

        elif mode.casefold() in {'rk', 'runge', 'kutta', 'rungekutta', 'runge-kutta'}:

            x_k1, y_k1, p_k1 = self.calculate_derivative(self.__fieldX, self.__fieldY, self.__fieldP)
            x_k2, y_k2, p_k2 = self.calculate_derivative(self.__fieldX + ((dt / 2) * x_k1), self.__fieldY + ((dt / 2) * y_k1), self.__fieldP + ((dt / 2) * p_k1))
            x_k3, y_k3, p_k3 = self.calculate_derivative(self.__fieldX + ((dt / 2) * x_k2), self.__fieldY + ((dt / 2) * y_k2), self.__fieldP + ((dt / 2) * p_k2))
            x_k4, y_k4, p_k4 = self.calculate_derivative(self.__fieldX + (dt * x_k3), self.__fieldY + (dt * y_k3), self.__fieldP + (dt * p_k3))

            self.__fieldX += (x_k1 + (2 * x_k2) + (2 * x_k3) + x_k4) * (dt / 6)
            self.__fieldY += (y_k1 + (2 * y_k2) + (2 * y_k3) + y_k4) * (dt / 6)
            self.__fieldP += (p_k1 + (2 * p_k2) + (2 * p_k3) + p_k4) * (dt / 6)

        else:
            raise TypeError('Unrecognized integration mode!')

        if not self.periodic:
            self.__fieldP[0][:] = 1.0
            self.__fieldP[-1][:] = 1.0
            self.__fieldP[:][0] = 1.0
            self.__fieldP[:][-1] = 1.0

    #------------------------------------------------------------------------------------------------
    # Returns the three time derivatives (dux/dt, duy/dt, dp/dt) for the supplied fields, which must
    # be supplied as numpy arrays. SHOULD NOT BE USED EXTERNALLY!
    
    def calculate_derivative(self, ux, uy, p) -> tuple[NDArray[Any], ...]:

        padtype = ('wrap' if self.periodic else 'edge')

        padded_field = np.pad(ux, (1, 1), mode=padtype)
        gradients = np.gradient(padded_field, self.res[0], self.res[1])
        duxdx = gradients[0][1:-1,1:-1]
        duxdy = gradients[1][1:-1,1:-1]
        
        padded_field = np.pad(uy, (1, 1), mode=padtype)
        gradients = np.gradient(padded_field, self.res[0], self.res[1])
        duydx = gradients[0][1:-1,1:-1]
        duydy = gradients[1][1:-1,1:-1]
        
        padded_field = np.pad(p, (1, 1), mode=padtype)
        gradients = np.gradient(padded_field, self.res[0], self.res[1])
        dpdx = gradients[0][1:-1,1:-1]
        dpdy = gradients[1][1:-1,1:-1]
        
        # Prevent division by zero, if pressure drops impossibly low
        rho = np.power(p, self.alpha, out=np.full_like(p, PRES_MIN), where=(p>PRES_MIN))
        
        duxdt = -(dpdx / rho) - (ux * duxdx) - (uy * duxdy)
        duydt = -(dpdy / rho) - (ux * duydx) - (uy * duydy)
        dpdt = -(p * (duxdx + duydy) / self.alpha) - (ux * dpdx) - (uy * dpdy)

        return (duxdt, duydt, dpdt)



