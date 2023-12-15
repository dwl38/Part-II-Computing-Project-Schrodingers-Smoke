import typing as typ
import math
from numbers import Number

#====================================================================================================
# A set of common utility functions.
#====================================================================================================

# Global constants (DO NOT MODIFY IN PROGRAM!)

EPSILON: typ.Final[complex] = complex(0.01)
PRES_MIN: typ.Final[float] = 0.01
T = typ.TypeVar('T')

#----------------------------------------------------------------------------------------------------
# Checks if the input parameter is a scalar.

def isscalar(x: typ.Any) -> bool:
    return isinstance(x, Number)

#----------------------------------------------------------------------------------------------------
# Rounds off the input number to a relevant number of significant figures.

def round_sig(x: typ.SupportsRound, sig: int = 3):
    if isinstance(x, (int, float)):
        if x == 0:
            return 0
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
    elif isinstance(x, complex):
        return complex(round_sig(x.real, sig), round_sig(x.imag, sig))
    else:
        return round_sig(float(x), sig)
    
#----------------------------------------------------------------------------------------------------
# Coerces what should be a tuple of dtypes into what is definitely a tuple of dtypes.

def coerce_tuple(x, ndims: int = 2, dtype = float) -> tuple:
    if isinstance(x, typ.Iterable):
        output = [dtype()] * ndims
        for i in range(min(ndims, len(x))):
            output[i] = dtype(x[i])
        return tuple(output)
    return tuple([dtype(x)] * ndims)

#----------------------------------------------------------------------------------------------------
# Prints a nicely formatted progress bar.

def print_progress_bar(progress: float, start: float = 0, stop: float = 100) -> None:

    fraction = (progress - start) / (stop - start)
    fraction = min(max(fraction, 0.0), 1.0)

    n_segments = 30 # Hardcoded
    n_filled = int(n_segments * fraction + 0.5)

    text = ('    Calculating... |' + ('\u2588' * n_filled) + ('-' * (n_segments - n_filled))
            + ('| {}%.'.format(round(100.0 * fraction, 1))))
    print(text, end='\r')


