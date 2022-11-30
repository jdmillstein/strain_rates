#-*- coding: utf-8 -*-

# ALL OF THIS is either directly from or slightly modified from https://github.com/bryanvriel/iceutils
# Bryan made an excellent package and I urge you to clone his repo instead of using this!

import numpy as np
import matplotlib.pyplot as plt
import os
from osgeo import gdal
import math as m
import matplotlib.colors as colors
import numpy as np
from itertools import product
from functools import partial
from multiprocessing import Pool

from scipy.ndimage import sobel
from scipy.signal import medfilt2d

##################################################################################
############### Must haves for dealing with offset outtputs ######################
##################################################################################

def load_offset_velocity_from_ds(ds, band=1, rangePixel=2.33, azPixel=14.2, dt=6.0):
    
    # Read data into array
    off = ds.GetRasterBand(band).ReadAsArray()
    
    # Scale by pixel size and time separation in order to get velocity
    pixel_sizes = [14.2, 2.33] # [az_pixel, rg_pixel] in meters
    off_meters = off * pixel_sizes[band-1]
    off_vel = off_meters / dt  #m/6 day
    off_vel = off_vel * 365.25
    
    return off_vel

def extent_from_ds(ds):
    """
    Unpack geotransform of GDAL dataset in order to compute map extents.
    """
    # Get raster size
    Ny = ds.RasterYSize
    Nx = ds.RasterXSize
    
    # Unpack geotransform
    try:
        xstart, dx, _, ystart, _, dy = ds.GetGeoTransform()
    except AttributeError:
        ystart = xstart = 0.0
        dx = dy = 1.0
        
    # Compute and return extent
    xstop = xstart + (Nx - 1) * dx
    ystop = ystart + (Ny - 1) * dy
    return np.array([xstart, xstop, ystop, ystart])


##################################################################################
##################################################################################
############### Bryan Riel iceutils modifications for 6 day offsets  #############
##################################################################################

def compute_strains_np(vx, vy, dx=100, dy=-100, window_size=5,
                          grad_method='numpy', inpaint=True, rotate=True,**kwargs):   
    
    # Cache image shape
    Ny, Nx = vx.shape

    # Compute velocity gradients
    L12, L11 = gradient(vx, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs)
    L22, L21 = gradient(vy, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs)

    
    
    # Compute components of strain-rate tensor
    D = np.empty((2, 2, vx.size))
    D[0, 0, :] = 0.5 * (L11 + L11).ravel()
    D[0, 1, :] = 0.5 * (L12 + L21).ravel()
    D[1, 0, :] = 0.5 * (L21 + L12).ravel()
    D[1, 1, :] = 0.5 * (L22 + L22).ravel()

    # Compute pixel-dependent rotation tensor if requested
    if rotate:
        R = np.empty((2, 2, vx.size))
        theta = np.arctan2(vy, vx).ravel()
        R[0, 0, :] = np.cos(theta)
        R[0, 1, :] = np.sin(theta)
        R[1, 0, :] = -np.sin(theta)
        R[1, 1, :] = np.cos(theta)

        # Apply rotation tensor
        D = np.einsum('ijm,kjm->ikm', D, R)
        D = np.einsum('ijm,jkm->ikm', R, D)
        

    # Cache elements of strain-rate tensor for easier viewing
    D11 = D[0, 0, :]
    D12 = D[0, 1, :]
    D21 = D[1, 0, :]
    D22 = D[1, 1, :]
    
    # Normal strain rates
    exx = D11.reshape(Ny, Nx)
    eyy = D22.reshape(Ny, Nx)

    # Shear- same result as: e_xy_max = np.sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    trace = D11 + D22
    det = D11 * D22 - D12 * D21
    shear = np.sqrt(0.25 * trace**2 - det).reshape(Ny, Nx)

    # Compute scalar quantities from stress tensors
    dilatation = (L11 + L22).reshape(Ny, Nx)
    effective_strain = np.sqrt(L11**2 + L22**2 + 0.25 * (L12 + L21)**2 + L11 * L22).reshape(Ny, Nx)

    # Store strain components in dictionary
    strain_dict = {'e_xx': exx,
                   'e_yy': eyy,
                   'e_xy': shear,
                   'dilatation': dilatation,
                   'effective': effective_strain}

    # Return strain and stress dictionaries
    return strain_dict, D
    
    
###############################################################################################

def compute_strains(vx, vy, dx=43, dy=-43, window_size=5,
                          grad_method='sgolay', inpaint=True, rotate=True,**kwargs):   
##def compute_strains(vx, vy, dx=43.338220530301498, dy=-43.338220530301498,window_size=25,
 #                         grad_method='sgolay', inpaint=True, rotate=True,**kwargs):
    
    # Cache image shape
    Ny, Nx = vx.shape

    # Compute velocity gradients
    L12, L11 = gradient(vx, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs)
    L22, L21 = gradient(vy, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs)

    # Compute components of strain-rate tensor
    D = np.empty((2, 2, vx.size))
    D[0, 0, :] = 0.5 * (L11 + L11).ravel()
    D[0, 1, :] = 0.5 * (L12 + L21).ravel()
    D[1, 0, :] = 0.5 * (L21 + L12).ravel()
    D[1, 1, :] = 0.5 * (L22 + L22).ravel()

    # Compute pixel-dependent rotation tensor if requested
    if rotate:
        R = np.empty((2, 2, vx.size))
        theta = np.arctan2(vy, vx).ravel()
        R[0, 0, :] = np.cos(theta)
        R[0, 1, :] = np.sin(theta)
        R[1, 0, :] = -np.sin(theta)
        R[1, 1, :] = np.cos(theta)

        # Apply rotation tensor
        D = np.einsum('ijm,kjm->ikm', D, R)
        D = np.einsum('ijm,jkm->ikm', R, D)

    # Cache elements of strain-rate tensor for easier viewing
    D11 = D[0, 0, :]
    D12 = D[0, 1, :]
    D21 = D[1, 0, :]
    D22 = D[1, 1, :]

    # Normal strain rates
    exx = D11.reshape(Ny, Nx)
    eyy = D22.reshape(Ny, Nx)

    # Shear- same result as: e_xy_max = np.sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    trace = D11 + D22
    det = D11 * D22 - D12 * D21
    shear = np.sqrt(0.25 * trace**2 - det).reshape(Ny, Nx)

    # Compute scalar quantities from stress tensors
    dilatation = (L11 + L22).reshape(Ny, Nx)
    effective_strain = np.sqrt(L11**2 + L22**2 + 0.25 * (L12 + L21)**2 + L11 * L22).reshape(Ny, Nx)

    # Store strain components in dictionary
    strain_dict = {'e_xx': exx,
                   'e_yy': eyy,
                   'e_xy': shear,
                   'dilatation': dilatation,
                   'effective': effective_strain}

    # Return strain and stress dictionaries
    return strain_dict, D

##########################################################################################

def gradient(z, spacing=1.0, axis=None, remask=True, method='sgolay',
             inpaint=False, **kwargs):
    """
    Calls either Numpy or Savitzky-Golay gradient computation routines.
    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple, spacing corresponds
        to axes directions specified by axis. Default: 1.0.
    axis: None or int or tuple of ints, optional
        Axis or axes to compute gradient. If None, derivative computed along all
        dimensions. Default: 0.
    remask: bool, optional
        Apply NaN mask on gradients. Default: True.
    method: str, optional
        Method specifier in ('numpy', 'sgolay', 'robust'). Default: 'numpy'.
    inpaint: bool, optional
        Inpaint image prior to gradient computation (recommended for
        'sgolay' method). Default: False.
    **kwargs:
        Extra keyword arguments to pass to specific gradient computation.
    Returns
    -------
    s: ndarray or list of ndarray
        Set of ndarrays (or single ndarry for only one axis) with same shape as z
        corresponding to the derivatives of z with respect to each axis.
    """
    # Mask mask of NaNs
    nan_mask = np.isnan(z)
    have_nan = np.any(nan_mask)

    # For sgolay method, we need to inpaint if NaNs detected
    if inpaint and have_nan:
        z_inp = _inpaint(z, mask=nan_mask)
    else:
        z_inp = z

    # Compute gradient with numpy
    if method == 'numpy':
        if isinstance(spacing, (tuple, list)):
            s = np.gradient(z_inp, spacing[0], spacing[1], axis=(0, 1), edge_order=2)
        else:
            s = np.gradient(z_inp, spacing, axis=axis, edge_order=2)

    # With Savtizky-Golay
    elif method == 'sgolay':
        s = sgolay_gradient(z_inp, spacing=spacing, axis=axis, **kwargs)

    # With robust polynomial
    elif method in ('robust_l2', 'robust_lp'):
        zs, z_dy, z_dx = robust_gradient(z_inp, spacing=spacing, lsq_method=method, **kwargs)
        s = (z_dy, z_dx)
        if axis is not None and isinstance(axis, int):
            s = s[axis]

    else:
        raise ValueError('Unsupported gradient method.')

    # Re-apply mask
    if remask and have_nan:
        if isinstance(s, (tuple, list)):
            for arr in s:
                arr[nan_mask] = np.nan
        else:
            s[nan_mask] = np.nan

    return s

#################################################################################

def sgolay_gradient(z, spacing=1.0, axis=None, window_size=5, order=2):
    """
    Wrapper around Savitzky-Golay code to compute window size in pixels and call _sgolay2d
    with correct arguments.
    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing is
        specified as (dy, dx) and derivative computed along both dimensions.
        Default is unitary spacing.
    axis: None or int, optional
        Axis along which to compute gradients. If None, gradient computed
        along both dimensions. Default: None.
    window_size: scalar or tuple of scalars, optional
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x). Default: 3.
    order: int, optional
        Polynomial order. Default: 4.
    Returns
    -------
    gradient: a
        Array or tuple of array corresponding to gradients. If both axes directions
        are specified, returns (dz/dy, dz/dx).
    """
    # Compute derivatives in both directions
    if axis is None or isinstance(spacing, (tuple, list)):

        # Compute window sizes
        wy, wx = compute_windows(window_size, spacing)

        # Unpack spacing
        dy, dx = spacing
        
        # Call Savitzky-Golay twice in order to use different window sizes
        sy = _sgolay2d(z, window_size, order=order, derivative='col')
        sx = _sgolay2d(z, window_size, order=order, derivative='row')

        # Scale by spacing and return
        return sy / dy, sx / dx

    # Or derivative in a single direction
    else:

        assert axis is not None, 'Must specify axis direction.'

        # Compute window size
        w = int(np.ceil(abs(window_size / spacing)))
        if w % 2 == 0:
            w += 1

        # Call Savitzky-Golay
        if axis == 0:
            s = _sgolay2d(z, w, order=order, derivative='col')
        elif axis == 1:
            s = _sgolay2d(z, w, order=order, derivative='row')
        else:
            raise ValueError('Axis must be 0 or 1.')

        # Scale by spacing and return
        return s / spacing

#################################################################################
    
def _sgolay2d(z, window_size=5, order=2, derivative=None):
    """
    Max Filter, January 2021.
    Original lower-level code from Scipy cookbook, with modifications to
    padding.
    """
    from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    Z = np.pad(z, half_size, mode='reflect')

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        Zf = fftconvolve(Z, m, mode='valid')
        return Zf
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        return Zc
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr, Zc

#################################################################################

def compute_windows(window_size, spacing):
    """
    Convenience function to compute window sizes in pixels given spacing of
    pixels in physical coordinates.
    Parameters
    ----------
    spacing: scalar or tuple of scalars
        Spacing between pixels along axis. If tuple, each element specifies
        spacing along different axes.
    window_size: scalar or tuple of scalars
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x).
    Returns
    -------
    w: tuple of scalars
        Odd-number window sizes in both axes directions in number of pixels.
    """
    # Unpack spacing
    if isinstance(spacing, (tuple, list)):
        assert len(spacing) == 2, 'Spacing must be 2-element tuple.'
        dy, dx = spacing
    else:
        dy = dx = spacing

    # Compute window sizes
    if isinstance(window_size, (tuple, list)):
        assert len(window_size) == 2, 'Window size must be 2-element tuple.'
        wy, wx = window_size
    else:
        wy = wx = window_size

    # Array of windows
    if isinstance(wy, np.ndarray):
        wy = np.ceil(np.abs(wy / dy)).astype(int)
        wx = np.ceil(np.abs(wx / dx)).astype(int)
        wy[(wy % 2) == 0] += 1
        wx[(wx % 2) == 0] += 1

    # Or scalar windows
    else:
        wy, wx = int(np.ceil(abs(wy / dy))), int(np.ceil(abs(wx / dx)))
        wy = _make_odd(wy)
        wx = _make_odd(wx)

    return wy, wx


#################################################################################

def inpaint(raster, mask=None, method='spring', r=3.0):
    """
    Inpaint a raster at NaN values or with an input mask.
    Parameters
    ----------
    raster: Raster or ndarray
        Input raster or array object to inpaint.
    mask: None or ndarry, optional
        Mask with same shape as raster specifying pixels to inpaint. If None,
        mask computed from NaN values. Default: None.
    method: str, optional
        Inpainting method from ('telea', 'biharmonic'). Default: 'telea'.
    r: scalar, optional
        Radius in pixels of neighborhood for OpenCV inpainting. Default: 3.0.
    Returns
    -------
    out_raster: Raster
        Output raster object.
    """
    if isinstance(raster, Raster):
        rdata = raster.data
    else:
        rdata = raster

    # Create mask
    if mask is None:
        mask = np.isnan(rdata)
    else:
        assert mask.shape == rdata.shape, 'Mask and raster shape mismatch.'

    # Check suitability of inpainting method with available packages
    if method == 'telea' and cv is None:
        warnings.warn('OpenCV package cv2 not found; falling back to spring inpainting.')
        method = 'spring'

    # Call inpainting
    if method == 'spring':
        inpainted = _inpaint_spring(rdata, mask)
    elif method == 'telea': 
        umask = mask.astype(np.uint8)
        inpainted = cv.inpaint(rdata, umask, r, cv.INPAINT_TELEA)
    elif method == 'biharmonic': 
        inpainted = inpaint_biharmonic(rdata, mask, multichannel=False)
    else:
        raise ValueError('Unsupported inpainting method.')

    # Return new raster or array
    if isinstance(raster, Raster):
        return Raster(data=inpainted, hdr=raster.hdr)
    else:
        return inpainted
    
    
def _make_odd(w):
    """
    Convenience function to ensure a number is odd.
    """
    if w % 2 == 0:
        w += 1
    return w

# Lambda for getting order-dependent polynomial exponents
_compute_exps = lambda order: [(k-n, n) for k in range(order + 1) for n in range(k + 1)]



# end of file