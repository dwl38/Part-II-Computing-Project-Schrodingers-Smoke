from __future__ import annotations
import numpy as np
from math import sqrt
from .common import *

#====================================================================================================
### DEPRECATED: No longer in use! See end comment. ###
#====================================================================================================
# A two-component complex vector, useful for e.g. the value of the wavefunction at a specific point.
# Note that each arithmetic operator has been overloaded in order to provide a sensible meaning:
#
#     - Addition or subtraction with another vector gives a vector; it is not possible to add or
#       subtract with a scalar or other generic object
#     - Multiplication with another CVec2D gives the Hermitian product, a complex scalar in general
#     - Multiplication or division with a scalar gives the expected result of scalar multiplication
#
# This class was designed to override the arithmetic operations used by numpy's broadcasting, such
# that an ndarray of CVec2Ds will operate in an intuitive manner. For example, contrast the following
# representations of a two-component complex wavefunction over three dimensions:
#
#     - An ndarray of shape (X, Y, Z) and dtype 'object', populated with CVec2D objects
#     - An ndarray of shape (X, Y, Z, 2) and dtype 'complex', populated with complex scalars
#
# Both representations are valid, but differ in ease of use and efficiency. For example, in order to
# generate the complex scalar field <phi|psi> (which should be an ndarray of shape (X, Y, Z) and
# dtype 'complex'), we can simply execute phi * psi in the first representation, whereas the best-
# practice of the second representation is numpy.einsum('ijkl,ijkl->ijk', numpy.conj(phi), psi). It
# should be noted that the second representation is more efficient, but its clumsiness of notation is
# potentially error-prone.
#====================================================================================================
### DEPRECATED: Since the wavefunctions in Schrodinger's Smoke have fixed shapes, better efficiency
###   can be achieved by simply storing two separate ndarrays and 'hardcoding' each operation. This
###   CVec2D structure may be useful for generalized behaviour, but this is currently unnecessary.
#====================================================================================================

class CVec2D:

    # Instance initializer; a CVec2D can be initialized by any of the following inputs:
    #
    #     - No arguments:          CVec2D() creates the zero vector (0, 0)
    #     - One scalar argument:   CVec2D(a) creates a single-component vector (a, 0)
    #     - Many scalar arguments: CVec2D(a, b, c, ...) creates the two-component vector (a, b) from
    #                                the first two arguments
    #     - One iterable argument: CVec2D((a, b, ...)) creates the two-component vector (a, b) from
    #                                the first two elements of the iterable container
    #     - An existing CVec2D:    CVec2D(v) creates a shallow copy of v

    def __init__(self, *args) -> None:
        self.a = complex(0, 0)
        self.b = complex(0, 0)
        if len(args) > 0:
            source = np.ravel(args)
            self.a = complex(source[0])
            self.b = complex(source[1]) if source.size > 1 else complex(0, 0)
    
    #------------------------------------------------------------------------------------------------
    # Returns the (squared or unsquared) magnitude of this vector; always non-negative real

    def mag_sq(self) -> float:
        return ((abs(self.a))**2 + (abs(self.b))**2)

    def mag(self) -> float:
        return sqrt((abs(self.a))**2 + (abs(self.b))**2)

    #------------------------------------------------------------------------------------------------
    # String representations

    def __repr__(self) -> str:
        return f'({self.a}, {self.b})'

    def __str__(self) -> str:
        return f'({round_sig(self.a)}, {round_sig(self.b)})'

    #------------------------------------------------------------------------------------------------
    # Equality comparisons

    def __eq__(self, other: CVec2D) -> bool:
        return isinstance(other, CVec2D) and (self.a == other.a) and (self.b == other.b)
    
    def __ne__(self, other: CVec2D) -> bool:
        return (not isinstance(other, CVec2D)) or (self.a != other.a) or (self.b != other.b)
    
    #------------------------------------------------------------------------------------------------
    # Hashing (e.g. for use in dicts)

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    #------------------------------------------------------------------------------------------------
    # Emulating a list of length 2; does not support negative indices

    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: typ.SupportsIndex) -> complex:
        index = key.__index__()
        if index == NotImplemented:
            raise TypeError(f'{type(key)} is not an index!')
        elif index == 0:
            return self.a
        elif index == 1:
            return self.b
        else:
            raise IndexError(f'{key} is out of bounds!')

    def __setitem__(self, key: typ.SupportsIndex, value: typ.SupportsComplex) -> complex:
        index = key.__index__()
        if index == NotImplemented:
            raise TypeError(f'{type(key)} is not an index!')
        elif index == 0:
            self.a = complex(value)
        elif index == 1:
            self.b = complex(value)
        else:
            raise IndexError(f'{key} is out of bounds!')

    def __contains__(self, item: typ.SupportsComplex) -> bool:
        test = complex(item)
        return (self.a == test or self.b == test)
    
    #------------------------------------------------------------------------------------------------
    # Vector addition

    def __add__(self, other: CVec2D) -> CVec2D:
        if isinstance(other, CVec2D):
            return CVec2D(self.a + other.a, self.b + other.b)
        else:
            return NotImplemented
    
    #------------------------------------------------------------------------------------------------
    # Vector subtraction

    def __sub__(self, other: CVec2D) -> CVec2D:
        if isinstance(other, CVec2D):
            return CVec2D(self.a - other.a, self.b - other.b)
        else:
            return NotImplemented
    
    #------------------------------------------------------------------------------------------------
    # Either scalar multiplication or Hermitian product

    @typ.overload
    def __mul__(self, other: Number) -> CVec2D:
        ...

    @typ.overload
    def __mul__(self, other: CVec2D) -> complex:
        ...

    def __mul__(self, other: Number | CVec2D) -> CVec2D | complex:
        if isinstance(other, CVec2D):
            return ((self.a.conjugate() * other.a) + (self.b.conjugate() * other.b))
        elif isscalar(other):
            return CVec2D(self.a * other, self.b * other)
        else:
            return NotImplemented
        
    #------------------------------------------------------------------------------------------------
    # Scalar division
    
    def __truediv__(self, other: Number) -> CVec2D:
        if isscalar(other):
            return CVec2D(self.a / other, self.b / other)
        else:
            return NotImplemented
        
    #------------------------------------------------------------------------------------------------
    # Vector addition (reversed)

    def __radd__(self, other: CVec2D) -> CVec2D:
        if isinstance(other, CVec2D):
            return CVec2D(self.a + other.a, self.b + other.b)
        else:
            return NotImplemented
    
    #------------------------------------------------------------------------------------------------
    # Vector subtraction (reversed)

    def __rsub__(self, other: CVec2D) -> CVec2D:
        if isinstance(other, CVec2D):
            return CVec2D(other.a - self.a, other.b - self.b)
        else:
            return NotImplemented
    
    #------------------------------------------------------------------------------------------------
    # Either scalar multiplication or Hermitian product (reversed)

    @typ.overload
    def __rmul__(self, other: Number) -> CVec2D:
        ...

    @typ.overload
    def __rmul__(self, other: CVec2D) -> complex:
        ...

    def __rmul__(self, other: Number | CVec2D) -> CVec2D | complex:
        if isinstance(other, CVec2D):
            return ((other.a.conjugate() * self.a) + (other.b.conjugate() * self.b))
        elif isscalar(other):
            return CVec2D(self.a * other, self.b * other)
        else:
            return NotImplemented
        
    #------------------------------------------------------------------------------------------------
    # In-place vector addition +=

    def __iadd__(self, other: CVec2D) -> CVec2D:
        if isinstance(other, CVec2D):
            self.a += other.a
            self.b += other.b
            return self
        else:
            return NotImplemented
        
    #------------------------------------------------------------------------------------------------
    # In-place vector subtraction -=

    def __isub__(self, other: CVec2D) -> CVec2D:
        if isinstance(other, CVec2D):
            self.a -= other.a
            self.b -= other.b
            return self
        else:
            return NotImplemented
        
    #------------------------------------------------------------------------------------------------
    # In-place scalar multiplication *=

    def __imul__(self, other: Number) -> CVec2D:
        if isscalar(other):
            self.a *= other
            self.b *= other
            return self
        else:
            return NotImplemented
        
    #------------------------------------------------------------------------------------------------
    # In-place scalar division /=

    def __itruediv__(self, other: Number) -> CVec2D:
        if isscalar(other):
            self.a /= other
            self.b /= other
            return self
        else:
            return NotImplemented
        
    #------------------------------------------------------------------------------------------------
    # Unary arithmetic operations (note that +v creates a copy of v)

    def __neg__(self) -> CVec2D:
        return CVec2D(-self.a, -self.b)
    
    def __pos__(self) -> CVec2D:
        return CVec2D(self.a, self.b)

    def __abs__(self) -> float:
        return self.mag()


