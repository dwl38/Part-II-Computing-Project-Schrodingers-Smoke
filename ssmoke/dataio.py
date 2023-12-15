import numpy as np
import os
from shutil import rmtree
from typing import Any, Iterable
from numpy.typing import NDArray
from .common import *

#====================================================================================================
# An interface for reading/writing data describing a fluid flow field, either in 2D or 3D, into a
# custom file format; each .data file is actually a directory, containing multiple compressed .npz
# archives (one 'header' and several 'segments') consisting of the following nparrays:
# 
#     header:
#     -------
#         Ndim:         size 1, containing the number of dimensions; expect Ndim = 2 or 3 only.
#         Nframes:      size 1, containing the total number of saved timeframes.
#         segNames:     size Nseg, containing the names of each data segment.
#         shape:        size Ndim, containing the shape of the spatial lattice (Nx, Ny, (Nz)).
#         res:          size Ndim, containing the resolution of the spatial lattice (dx, dy, (dz)).
#         cx, cy, (cz): size (Nx, Ny, (Nz)), containing the x and y (and z) coordinates of the
#                         lattice in meshgrid-style.
# 
#     segments:
#     ---------
#         times:        total size Nframes, containing the t values of each timeframe.
#         vx, vy, (vz): total size (Nframes, Nx, Ny, (Nz)), containing the x and y (and z) components
#                         of the fluid flow velocity at each timeframe at every point of the lattice.
# 
# The specifications above dictate the contents of the on-harddrive file (which is divided into
# segments in order to circumvent GitHub's maximum filesize constraint); however, the output of the
# load_data() method automatically consolidates the segments together, thus the in-memory loaded file
# is not segmented and does not have the segNames field.
#====================================================================================================



#----------------------------------------------------------------------------------------------------
# Read a flow field (2D or 3D) from the provided filename; returns a dictionary whose key-value pairs
# are the names of the datafields listed above and the relevant nparrays.
# 
# Note that this dictionary represents the 'consolidated' data, hence the output has the 'times' and
# 'vx' etc. fields consolidated into nparrays of length Nframes, and the 'segNames' field does not
# exist.

def load_data(filename: str) -> dict:

    parentdir = os.path.join(os.getcwd(), (filename if filename.endswith('.data') else (filename + '.data')))
    output = dict()

    header = np.load(os.path.join(parentdir, 'header.npz'))
    output.update(Ndim=header['Ndim'])
    output.update(Nframes=header['Nframes'])
    output.update(shape=header['shape'])
    output.update(res=header['res'])
    output.update(cx=header['cx'])
    output.update(cy=header['cy'])
    if output['Ndim'] == 3:
        output.update(cz=header['cz'])

    segNames = header['segNames']
    times = []
    vx = []
    vy = []
    if output['Ndim'] == 3:
        vz = []

    for name in segNames:
        segment = np.load(os.path.join(parentdir, name))
        times.append(segment['times'])
        vx.append(segment['vx'])
        vy.append(segment['vy'])
        if output['Ndim'] == 3:
            vz.append(segment['vz'])

    output.update(times=np.concatenate(times, axis=0))
    output.update(vx=np.concatenate(vx, axis=0))
    output.update(vy=np.concatenate(vy, axis=0))
    if output['Ndim'] == 3:
        output.update(vz=np.concatenate(vz, axis=0))

    return output

#----------------------------------------------------------------------------------------------------
# Save a 2D flow field; the following parameters must be provided:
#
#     - Nframes: positive integer
#     - shape:   tuple or otherwise iterable, containing (Nx, Ny)
#     - res:     tuple or otherwise iterable, containing (dx, dy)
#     - times:   list or otherwise iterable of length Nframes, containing scalars t
#     - xcoords: nparray of shape (Nx, Ny)
#     - ycoords: nparray of shape (Nx, Ny)
#     - vx:      list or otherwise iterable of length Nframes, containing nparrays of shape (Nx, Ny)
#     - vy:      list or otherwise iterable of length Nframes, containing nparrays of shape (Nx, Ny)
#
# This function will overwrite any existing data of the same filename.

def save_2d(filename: str, Nframes: int, shape: Iterable[int], res: Iterable[float],
            times: Iterable[float], xcoords: NDArray[Any], ycoords: NDArray[Any],
            vx: Iterable[NDArray[Any]], vy: Iterable[NDArray[Any]]) -> None:
    
    _Nframes = int(Nframes)
    _shape = np.array(coerce_tuple(shape, 2, int), dtype=int)
    _res = np.array(coerce_tuple(res, 2, float), dtype=float)
    _times = np.array(coerce_tuple(times, _Nframes, float), dtype=float)
    _cx = np.array(xcoords)
    _cy = np.array(ycoords)

    if len(vx) != _Nframes or len(vy) != _Nframes:
        raise TypeError('The input velocities do not have the correct number of timeframes!')
    _vx = np.array(vx)
    _vy = np.array(vy)

    parentdir = os.path.join(os.getcwd(), (filename if filename.endswith('.data') else (filename + '.data')))
    if os.path.exists(parentdir):
        try:
            rmtree(parentdir)
        except OSError as e:
            raise OSError(f'Error occurred while overwriting data; filename <{e.filename}>, original message <{e.strerror}>.')
    os.makedirs(parentdir)

    data_size_per_frame = 2 * _shape.prod() * _vx.itemsize
    frames_per_segment = max(1, int(20000000 / data_size_per_frame)) # Each segment is ~20MB

    if _Nframes % frames_per_segment == 0:
        Nseg = _Nframes // frames_per_segment
    else:
        Nseg = int(_Nframes // frames_per_segment) + 1
    segNames = np.array([('segment' + str(i) + '.npz') for i in range(Nseg)])

    np.savez_compressed(os.path.join(parentdir, 'header.npz'), Ndim=2, Nframes=_Nframes, segNames=segNames, shape=_shape, res=_res, cx=_cx, cy=_cy)
    for i in range(Nseg):
        start = i * frames_per_segment
        end = min((i + 1) * frames_per_segment, _Nframes)
        np.savez_compressed(os.path.join(parentdir, segNames[i]), times=_times[start:end], vx=_vx[start:end], vy=_vy[start:end])
        
#----------------------------------------------------------------------------------------------------
# Save a 3D flow field; the following parameters must be provided:
#
#     - Nframes: positive integer
#     - shape:   tuple or otherwise iterable, containing (Nx, Ny, Nz)
#     - res:     tuple or otherwise iterable, containing (dx, dy, dz)
#     - times:   list or otherwise iterable of length Nframes, containing scalars t
#     - xcoords: nparray of shape (Nx, Ny, Nz)
#     - ycoords: nparray of shape (Nx, Ny, Nz)
#     - zcoords: nparray of shape (Nx, Ny, Nz)
#     - vx:      list or otherwise iterable of length Nframes, containing nparrays of shape (Nx, Ny, Nz)
#     - vy:      list or otherwise iterable of length Nframes, containing nparrays of shape (Nx, Ny, Nz)
#     - vz:      list or otherwise iterable of length Nframes, containing nparrays of shape (Nx, Ny, Nz)
#
# This function will overwrite any existing data of the same filename.

def save_3d(filename: str, Nframes: int, shape: Iterable[int], res: Iterable[float],
            times: Iterable[float], xcoords: NDArray[Any], ycoords: NDArray[Any],
            zcoords: NDArray[Any], vx: Iterable[NDArray[Any]], vy: Iterable[NDArray[Any]],
            vz: Iterable[NDArray[Any]]) -> None:
    
    _Nframes = int(Nframes)
    _shape = np.array(coerce_tuple(shape, 3, int), dtype=int)
    _res = np.array(coerce_tuple(res, 3, float), dtype=float)
    _times = np.array(coerce_tuple(times, _Nframes, float), dtype=float)
    _cx = np.array(xcoords)
    _cy = np.array(ycoords)
    _cz = np.array(zcoords)

    if len(vx) != _Nframes or len(vy) != _Nframes or len(vz) != _Nframes:
        raise TypeError('The input velocities do not have the correct number of timeframes!')
    _vx = np.array(vx)
    _vy = np.array(vy)
    _vz = np.array(vz)

    parentdir = os.path.join(os.getcwd(), (filename if filename.endswith('.data') else (filename + '.data')))
    if os.path.exists(parentdir):
        try:
            rmtree(parentdir)
        except OSError as e:
            raise OSError(f'Error occurred while overwriting data; filename <{e.filename}>, original message <{e.strerror}>.')
    os.makedirs(parentdir)

    data_size_per_frame = 3 * _shape.prod() * _vx.itemsize
    frames_per_segment = max(1, int(20000000 / data_size_per_frame)) # Each segment is ~20MB

    if _Nframes % frames_per_segment == 0:
        Nseg = _Nframes // frames_per_segment
    else:
        Nseg = int(_Nframes // frames_per_segment) + 1
    segNames = np.array([('segment' + str(i) + '.npz') for i in range(Nseg)])

    np.savez_compressed(os.path.join(parentdir, 'header.npz'), Ndim=3, Nframes=_Nframes, segNames=segNames, shape=_shape, res=_res, cx=_cx, cy=_cy, cz=_cz)
    for i in range(Nseg):
        start = i * frames_per_segment
        end = min((i + 1) * frames_per_segment, _Nframes)
        np.savez_compressed(os.path.join(parentdir, segNames[i]), times=_times[start:end], vx=_vx[start:end], vy=_vy[start:end], vz=_vz[start:end])
        
#----------------------------------------------------------------------------------------------------
# Deletes a file. Returns true if the operation was successful, and false if the specified file does
# not exist or if an exception occurs during deletion.

def delete_data(filename: str) -> bool:

    parentdir = os.path.join(os.getcwd(), (filename if filename.endswith('.data') else (filename + '.data')))
    if os.path.exists(parentdir):
        try:
            rmtree(parentdir)
            return True
        except OSError as e:
            return False
    return False
