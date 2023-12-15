from __future__ import annotations
import numpy as np
import scipy.fft as fft
import math
from typing import Any, Callable, Iterable
from numpy.typing import ArrayLike, NDArray
from .cvec2d import CVec2D
from .common import *

#====================================================================================================
# The SSmoke2D class represents a single 'timestate' of a Schrodingerized flow field, i.e. contains
# all of the necessary information to describe a incompressible and inviscid velocity field over a
# two-dimensional rectangular domain, as well as to propagate it into the next timestep using the
# Schrodinger's Smoke algorithm. An instance of the SSmoke2D class stores the following information:
#
#     - Domain information:
#         - Two integers (Nx, Ny) denoting the number of spatial 'gridcells' used in each dimension.
#         - Two floats (dx, dy) denoting the size of each gridcell in each dimension.
#
#     - Wavefunction information:
#         - One float hbar controlling the vorticity element size.
#         - Two numpy.ndarrays of complex numbers, representing the first and second components of
#             the value of the wavefunction (which is, mathematically speaking, a map from R2 to C2)
#             at the centre of each gridcell.
#
#     - Constraint information [optional]:
#         - A list of VelocityConstraint2D objects, each representing the enforcement of a fixed
#             velocity in some region; this may be updated between timesteps. A stationary obstacle 
#             can be represented by a constant region of zero velocity, for example, while a region
#             of positive velocity can be used to simulate a jet.
#
# Notationally, a distinction is made between the 'real space' comprised of continuous coordinates
# (x, y) and the discrete 'lattice' of gridcells specified by integers (i, j); all functions' names
# adhere to this distinction.
#
# Periodic boundary conditions are assumed unless otherwise specified. In order to simulate e.g. a
# closed box of size N, a SSmoke2D object of size N + 2 should be used, with obstacles placed at the
# walls. There is an optional parameter which allows for 'open' boundary conditions to be used
# instead, which loosely represents zero external pressure (free divergence at boundaries), but this
# may not exactly correspond to any physically meaningful boundary conditions.
#====================================================================================================

class SSmoke2D:
    
    # Instance initializer; a SSmoke2D can be initialized using the following parameters:
    #
    #     - lattice_size: By default, this should be a tuple of ints (Nx, Ny). If a single integer N
    #                       is supplied, this will be interpreted as (N, N). This parameter is
    #                       ignored if initValues is specified as a SSmoke2D object.
    #
    #     - resolution:   By default, this should be a tuple of floats (dx, dy). If a single float dx
    #                       is supplied, this will be interpreted as (dx, dx). This parameter is
    #                       ignored if initValues is specified as a SSmoke2D object.
    #
    #     - hbar:         A float hbar giving the vorticity resolution of this SSmoke2D object.
    #
    #     - periodic:     Whether the boundary conditions are periodic (true) or open (false).
    #
    #     - initValues:   Optional argument; it supplies the following behaviours:
    #                         - If unspecified, creates a flow field with zero velocity everywhere.
    #                         - If given as a numpy.ndarray of shape (2, Nx, Ny), copies the supplied
    #                             values directly into the C2 wavefunction.
    #                         - If given as a single-parameter function, which maps a tuple of floats
    #                             (x, y) to a tuple of floats (vx, vy), creates a flow field with
    #                             initial velocity distributed according to the supplied function.
    #                         - If given as a single-parameter function, which maps a tuple of floats
    #                             (x, y) to a CVec2D (psiA, psiB), creates a C2 wavefunction with the
    #                             values of the supplied function.
    #                         - If given as a SSmoke2D object, creates a shallow copy.
    #                     Providing any other type of input may result in exceptions being thrown.
    #                       The functional input types must also be well-defined at (0, 0).
    #
    #     - constraints:  Optional argument; if specified, this must be an iterable of velocity
    #                       constraints, each in the form of a VelocityConstraint2D object.

    def __init__(self, lattice_size: int | Iterable[int] = 1,
                 resolution: float | Iterable[float] = 1.0,
                 hbar: float = 0.02,
                 periodic: bool = True,
                 initValues: ArrayLike | Callable[[Iterable[float]], Any] | SSmoke2D = None,
                 constraints: Iterable[VelocityConstraint2D] = None) -> None:
        
        if isinstance(initValues, SSmoke2D):
            self.shape = initValues.shape                   # Parameter
            self.res = initValues.res                       # Parameter
            self.__multi_iter = initValues.__multi_iter     # Memoization, for internal optimization
            self.__ksq = initValues.__ksq                   # Memoization, for internal optimization
            self.__lplc_eigvals = initValues.__lplc_eigvals # Memoization, for internal optimization
            self.__xcoords = initValues.__xcoords           # Memoization...
            self.__ycoords = initValues.__ycoords           # Memoization...
            self.hbar = initValues.hbar                     # Parameter
            self.periodic = initValues.periodic             # Parameter
            self.__fieldA = initValues.__fieldA.copy()      # Parameter
            self.__fieldB = initValues.__fieldB.copy()      # Parameter
            self.__constraints = (None if (initValues.__constraints is None) else initValues.__constraints.copy())
        else:
            
            self.shape: tuple[int, int] = coerce_tuple(lattice_size, ndims=2, dtype=int)
            self.res: tuple[float, float] = coerce_tuple(resolution, ndims=2, dtype=float)
                
            self.__multi_iter = [(i,j) for i in range(self.shape[0]) for j in range(self.shape[1])]
            self.__ksq = None
            self.__lplc_eigvals = None
            self.__xcoords = None
            self.__ycoords = None
            self.hbar = float(hbar)
            self.periodic = bool(periodic)

            if initValues is None:
                self.__fieldA = np.ones(self.shape, dtype=complex)
                self.__fieldB = np.full(self.shape, EPSILON, dtype=complex)

            elif isinstance(initValues, Callable):

                test = initValues((0.0, 0.0))
                self.__fieldA = np.ones(self.shape, dtype=complex)
                self.__fieldB = np.full(self.shape, EPSILON, dtype=complex)

                if isinstance(test, Iterable) and isscalar(test[0]):
                    for i, j in self.__multi_iter:
                        x = i * self.res[0]
                        y = j * self.res[1]
                        vx, vy = initValues((x, y))
                        self.__fieldA[i][j] = np.exp(1j * ((vx * x) + (vy * y)) / self.hbar)
                elif isinstance(test, CVec2D):
                    for i, j in self.__multi_iter:
                        x = i * self.res[0]
                        y = j * self.res[1]
                        cv = initValues((x, y))
                        self.__fieldA[i][j] = cv[0]
                        self.__fieldB[i][j] = cv[1]
                else:
                    raise TypeError(r'Input initValues is a function of wrong output type!')

            elif isinstance(initValues, Iterable) and isscalar(initValues[0]):

                self.__fieldA = np.ones(self.shape, dtype=complex)
                self.__fieldB = np.full(self.shape, EPSILON, dtype=complex)
                for i, j in self.__multi_iter:
                    x = i * self.res[0] / hbar
                    y = j * self.res[1] / hbar
                    self.__fieldA[i][j] = np.exp(1j * ((initValues[0] * x) + (initValues[1] * y)))

            elif isinstance(initValues, np.ndarray):

                if initValues.shape == (2, self.shape[0], self.shape[1]):
                    self.__fieldA = initValues[0].astype(complex)
                    self.__fieldB = initValues[1].astype(complex)
                else:
                    raise IndexError(r'Input initValues is a ndarray of wrong shape!')

            else:
                raise TypeError(r'Input initValues is not recognized as a valid input type!')

            if constraints is None:
                self.__constraints = None
            elif isinstance(constraints, VelocityConstraint2D):
                if constraints.region.shape == self.shape:
                    self.__constraints = [constraints]
                else:
                    raise IndexError(r'Input constraint has the wrong shape!')
            elif isinstance(constraints, Iterable):
                self.__constraints = []
                for constraint in constraints:
                    if isinstance(constraint, VelocityConstraint2D) and constraint.region.shape == self.shape:
                        self.__constraints.append(constraint)
                if len(self.__constraints) == 0:
                    raise TypeError(r'Input constraints is an interable containing no valid constraints!')
            else:
                raise TypeError(r'Input constraints is not recognized as a valid input type!')

            if self.__constraints is not None:
                self.enforce_velocity_constraints()
    
    #------------------------------------------------------------------------------------------------
    # Returns the 1st or 2nd component of the C2 wavefunction as a numpy.ndarray; the components are
    # zero-indexed, hence 'index' should be either 0 or 1.

    def get_complex_field(self, index: int) -> NDArray[Any]:
        if index == 0:
            return self.__fieldA
        elif index == 1:
            return self.__fieldB
        raise IndexError(f'Out of bounds: {index} should be either 0 or 1!')
    
    #------------------------------------------------------------------------------------------------
    # Returns the partial derivatives of the 1st or 2nd component of the C2 wavefunction as a pair of
    # numpy.ndarrays; again 'index' should be 0 or 1.

    def get_field_gradient(self, index: int) -> tuple[NDArray[Any], NDArray[Any]]:

        if index == 0:
            field_to_diff = self.__fieldA
        elif index == 1:
            field_to_diff = self.__fieldB
        else:
            raise IndexError(f'Out of bounds: {index} should be either 0 or 1!')

        if self.periodic:
            padded_field = np.pad(field_to_diff, (1, 1), 'wrap')           # Padding the field to enforce PBCs
        else:
            padded_field = np.pad(field_to_diff, (1, 1), 'edge')           # ...or open BCs

        gradients = np.gradient(padded_field, self.res[0], self.res[1])    # But need to unpad this now
        return (gradients[0][1:-1,1:-1], gradients[1][1:-1,1:-1])

    #------------------------------------------------------------------------------------------------
    # Returns two 1D arrays representing the x and y coordinates of every gridcell; i.e. the numpy- 
    # style meshgrid over this domain. The intended notation is x,y = this.meshgrid().

    def meshgrid(self) -> list[NDArray[Any], NDArray[Any]]:

        if self.__xcoords is None:
            xcoords = [i * self.res[0] for i in range(self.shape[0])]
            ycoords = [j * self.res[1] for j in range(self.shape[1])]
            self.__xcoords, self.__ycoords = np.meshgrid(xcoords, ycoords, indexing='ij')
        
        return (self.__xcoords.copy(), self.__ycoords.copy())
    
    #------------------------------------------------------------------------------------------------
    # Returns two 1D arrays representing the vx and vy components of the flow velocity at every
    # gridcell, corresponding to the meshgrid. The intended notation is vx,vy = this.flow_vel().

    def flow_vel(self) -> tuple[NDArray[Any], NDArray[Any]]:
        gradAx, gradAy = self.get_field_gradient(0)
        gradBx, gradBy = self.get_field_gradient(1)
        vx = -self.hbar * np.imag(np.conj(gradAx) * self.__fieldA + np.conj(gradBx) * self.__fieldB)
        vy = -self.hbar * np.imag(np.conj(gradAy) * self.__fieldA + np.conj(gradBy) * self.__fieldB)
        return (vx, vy)

    #------------------------------------------------------------------------------------------------
    # Get a constraint, by reference; this can be used to modify a constraint.

    def constraint(self, index: int) -> VelocityConstraint2D:
        if self.__constraints is None:
            return None
        return self.__constraints[index]
        
    #------------------------------------------------------------------------------------------------
    # Advances the flow field by a timestep dt, using the Schrodinger's Smoke algorithm.

    def advance_timestep(self, dt: float) -> None:

        self.propagate_schrodinger(dt)

        mag = np.sqrt(np.square(np.abs(self.__fieldA)) + np.square(np.abs(self.__fieldB)))
        self.__fieldA /= mag
        self.__fieldB /= mag

        self.enforce_divergence_constraints()

        if self.__constraints is not None:
            self.enforce_velocity_constraints()

    #------------------------------------------------------------------------------------------------
    # Performs the spectral portion of the integrator, i.e. advances the linear Schrodinger term by
    # timestep dt without enforcing pressure term. SHOULD NOT BE USED EXTERNALLY!
    
    def propagate_schrodinger(self, dt: float) -> None:

        fieldAk = fft.fft2(self.__fieldA)
        fieldBk = fft.fft2(self.__fieldB)

        if self.__ksq is None:
            kx = fft.fftfreq(self.shape[0], self.res[0])
            ky = fft.fftfreq(self.shape[1], self.res[1])
            self.__ksq = np.empty(self.shape, dtype=float)
            for i, j in self.__multi_iter:
                self.__ksq[i][j] = (kx[i])**2 + (ky[j])**2

        propagator = -(4j) * ((np.pi)**2) * self.hbar * dt / 2
        fieldAk *= np.exp(self.__ksq * propagator)
        fieldBk *= np.exp(self.__ksq * propagator)

        self.__fieldA = fft.ifft2(fieldAk)
        self.__fieldB = fft.ifft2(fieldBk)
        
    #------------------------------------------------------------------------------------------------
    # Enforces divergence constraint div(v) = 0. SHOULD NOT BE USED EXTERNALLY!

    def enforce_divergence_constraints(self) -> None:

        divfieldr = np.zeros(self.shape, dtype=float)

        #1st neighbour: (i, j) -> (i + 1, j)
        rolledA = np.roll(self.__fieldA, 1, axis=0)
        rolledB = np.roll(self.__fieldB, 1, axis=0)
        eta = np.angle((np.conj(self.__fieldA) * rolledA) + (np.conj(self.__fieldB) * rolledB))
        divfieldr += eta / ((self.res[0])**2)
        
        #2nd neighbour: (i, j) -> (i - 1, j)
        rolledA = np.roll(self.__fieldA, -1, axis=0)
        rolledB = np.roll(self.__fieldB, -1, axis=0)
        eta = np.angle((np.conj(self.__fieldA) * rolledA) + (np.conj(self.__fieldB) * rolledB))
        divfieldr += eta / ((self.res[0])**2)
        
        #3rd neighbour: (i, j) -> (i, j + 1)
        rolledA = np.roll(self.__fieldA, 1, axis=1)
        rolledB = np.roll(self.__fieldB, 1, axis=1)
        eta = np.angle((np.conj(self.__fieldA) * rolledA) + (np.conj(self.__fieldB) * rolledB))
        divfieldr += eta / ((self.res[1])**2)
        
        #4th neighbour: (i, j) -> (i, j - 1)
        rolledA = np.roll(self.__fieldA, -1, axis=1)
        rolledB = np.roll(self.__fieldB, -1, axis=1)
        eta = np.angle((np.conj(self.__fieldA) * rolledA) + (np.conj(self.__fieldB) * rolledB))
        divfieldr += eta / ((self.res[1])**2)

        if self.__lplc_eigvals is None:
            self.__lplc_eigvals = np.empty(self.shape, dtype=float)
            for i, j in self.__multi_iter:
                self.__lplc_eigvals[i][j] = (-((2 * math.sin(np.pi * i / self.shape[0]) / self.res[0])**2)
                                             - ((2 * math.sin(np.pi * j / self.shape[1]) / self.res[1])**2))
            self.__lplc_eigvals[np.where(self.__lplc_eigvals == 0)] = np.Infinity
            
        divfieldk = fft.fft2(divfieldr)
        divfieldk /= self.__lplc_eigvals
        divfieldr = fft.ifft2(divfieldk)
        phase_shift = np.exp(-(1j) * divfieldr)
        
        if not self.periodic:
            phase_shift[0,:] = 1
            phase_shift[-1,:] = 1
            phase_shift[:,0] = 1
            phase_shift[:,-1] = 1


        self.__fieldA *= phase_shift
        self.__fieldB *= phase_shift

    #------------------------------------------------------------------------------------------------
    # Enforces velocity constraint v = 0 at obstacles. SHOULD NOT BE USED EXTERNALLY!

    def enforce_velocity_constraints(self) -> None:

        n_attempts = 10 # Hardcoded
        assert (self.__constraints is not None)

        if self.__xcoords is None:
            xcoords = [i * self.res[0] for i in range(self.shape[0])]
            ycoords = [j * self.res[1] for j in range(self.shape[1])]
            self.__xcoords, self.__ycoords = np.meshgrid(xcoords, ycoords, indexing='ij')

        for _ in range(n_attempts):
            for constraint in self.__constraints:
                vx, vy = constraint.velocity
                phase = np.exp(1j * ((vx * self.__xcoords) + (vy * self.__ycoords)) / self.hbar)
                self.__fieldA = np.multiply(np.abs(self.__fieldA), phase, out=self.__fieldA, where=constraint.region)
                self.__fieldB = np.multiply(np.abs(self.__fieldB), phase, out=self.__fieldB, where=constraint.region)
            self.enforce_divergence_constraints()



#====================================================================================================
# The VelocityConstraint2D class represents a velocity constraint for a SSmoke2D flow. Each member
# contains only two properties, both of which are externally accessible:
#
#     - region:   a numpy.ndarray of bools, with a value of True for cells which are within the
#                   region (i.e. constrained) and a value of False for cells which are outside.
# 
#     - velocity: a tuple of floats (vx, vy) denoting the velocity enforced inside the region.
#
# Really it is just a data structure for bundling these two properties together.
#====================================================================================================

class VelocityConstraint2D:

    def __init__(self, lattice_size: int | Iterable[int] = 1,
                 velocity: Iterable[float] = (0.0, 0.0),
                 region: NDArray[Any] = None) -> None:

        shape: tuple[int, int] = coerce_tuple(lattice_size, ndims=2, dtype=int)
        self.velocity: tuple[float, float] = coerce_tuple(velocity, ndims=2, dtype=float)

        if region is None:
            self.region = np.zeros(shape, dtype=bool)
        elif isinstance(region, np.ndarray):
            if region.shape != shape:
                raise IndexError(r'Input region is a ndarray of wrong shape!')
            self.region = region.astype(bool)
        elif isinstance(region, Iterable) and isinstance(region[0], Iterable) and isinstance(region[0][0], bool):
            self.region = np.array(region, dtype=bool)
            if self.region.shape != shape:
                raise IndexError(r'Input region is a nested iterable of wrong shape!')
        else:
            raise TypeError(r'Input region is not of a recognized type!')
