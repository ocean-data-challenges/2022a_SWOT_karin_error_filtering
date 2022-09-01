#!/usr/bin/env python
# coding: utf-8

"""

    Various 2D filters working on numpy arrays

"""

import numpy as np
import scipy.special as spec
from math import pi as pi
from scipy.ndimage import generic_filter
from scipy.signal import convolve2d
from numpy.linalg import pinv
#import numba as nb

def median_filter(img, size):
    ''' median filter based on scipy.ndimage.generic_filter '''
    mask = np.isnan(img)
    filtered = generic_filter(img,
                              np.nanmedian,
                              size=size)
    filtered[mask] = np.nan
    return filtered

def gaussian_filter(img, sigma):
    ''' gaussian filter '''
    kernel = gaussian_kernel_weights(sigma)
    mask = np.isnan(img)
    img[mask] = 0.
    indic = np.ones(np.shape(img), dtype=float)
    indic[mask] = 0.
    img_f = convolve2d(img, kernel, mode='same')
    indic_f = convolve2d(indic, kernel, mode='same')
    filtered = img_f / indic_f
    filtered[mask] = np.nan
    return filtered

def gaussian_kernel_weights(sigma):
    ''' compute gaussian kernel weights '''
    lx = np.ceil(3*sigma)
    x = np.arange(lx)
    xx,yy = np.meshgrid(x,x,indexing='ij')
    w = 1./(2*pi*sigma**2)*np.exp(-(xx**2+yy**2)/(2*sigma**2))
    return w

def boxcar_filter(img, lx):
    ''' boxcar (running mean) filter '''
    kernel = boxcar_kernel_weights(lx)
    mask = np.isnan(img)
    img[mask] = 0.
    indic = np.ones(np.shape(img), dtype=float)
    indic[mask] = 0.
    img_f = convolve2d(img, kernel, mode='same')
    indic_f = convolve2d(indic, kernel, mode='same')
    indic_f = np.clip(indic_f, 1e-8, None)
    filtered = img_f / indic_f
    filtered[mask] = np.nan
    return filtered

def boxcar_kernel_weights(lx):
    """ compute boxcar kernel weights """
    return np.ones((lx,lx))


def lee_filter(img, lx):
    """ filter following Lee, 1980 """
    # filtrage par moyenne glissante
    lf = boxcar_filter(img, lx)
    
    # anomalies HF
    hf = img - lf
    
    # niveau de bruit HF
    hf_noise = np.nanstd(hf)
    
    # niveau de bruit local
    q = boxcar_filter(np.square(hf), lx) - hf_noise**2
    
    # gain
    k = q/(q + hf_noise**2)
    
    # image filtree
    filtered = lf + np.multiply(k, hf)
    
    return filtered


def lanczos_filter(img, lx, width_factor=3):
    """ lanczos filter """
    # compute kernel weights
    kernel = lanczos_kernel_weights(lx, width_factor)

    # set undefined values to 0
    mask = np.isnan(img)
    img[mask] = 0.

    # create indicator of undefined values
    indic = np.ones(np.shape(img), dtype=float)
    indic[mask] = 0.

    # convolve with kernel
    img_f = convolve2d(img, kernel, mode='same')
    indic_f = convolve2d(indic, kernel, mode='same')

    # reapply mask abnd return
    filtered = img_f / indic_f
    filtered[mask] = np.nan
    return filtered

def lanczos_kernel_weights(lx, width_factor):
    ''' compute lanczos kernel weights '''
    hw = np.ceil(width_factor * lx).astype(int)
    fc = 1. / lx

    # compute kernel weights
    kx, ky = np.meshgrid(np.arange(-hw, hw + 1), np.arange(-hw, hw + 1))
    z = np.sqrt((fc * kx) ** 2 + (fc * ky) ** 2)
    w_rect = 1 / z * fc * fc * spec.j1(2 * pi * z)
    w = (w_rect
         * np.sin(pi * kx / hw) / (pi * kx / hw)
         * np.sin(pi * ky / hw) / (pi * ky / hw))

    # there is a singularity where z=0
    w[:, hw] = (w_rect[:, hw]
                * 1. / (pi * ky[:, hw] / hw)
                * np.sin(pi * ky[:, hw] / hw))
    w[hw, :] = (w_rect[hw, :]
                * 1. / (pi * kx[hw, :] / hw)
                * np.sin(pi * kx[hw, :] / hw))
    w[hw, hw] = pi * fc * fc

    # force axisymetry
    w[(ky / float(hw)) ** 2 + (kx / float(hw)) ** 2 > 1] = 0

    # normalization
    w = w / np.sum(w)
    return w

#@nb.njit()
def loess_filter(img, degree, length, kernel):
    """
    2d loess filter
    adapted from https://github.com/arokem/lowess/blob/master/lowess/lowess.py
    """

    if kernel not in ['tricube','gaussian','epanechnikov']:
        raise NotImplementedError()

    (nx, ny) = np.shape(img)

    xx, yy = loess_meshgrid(
        np.arange(nx),
        np.arange(ny)
    )
    x = np.vstack((xx.flatten(), yy.flatten()))
    y = img.flatten()

    filtered = np.zeros(x.shape[-1])

    # remove NaNs
    y0 = y[~np.isnan(y)]
    x0 = x[:, ~np.isnan(y)]

    # boucle sur les positions
    for ii, pos in enumerate(x.T):
        if np.isnan(y[ii]):
            filtered[ii] = np.nan
        else:
            # calcul des distances au point courant (pos)
            distance = loess_distance(pos, x0)
            # selection des mesures tq distance < l
            idx = loess_find_selection(distance, length)

            # application de la selection
            x_sel = loess_apply_selection(x0, idx)
            y_sel = loess_apply_selection(y0, idx)
            d_sel = loess_apply_selection(distance, idx)

            # calcul des poids des obs
            w = loess_weights(d_sel, kernel)

            # inversion du système local
            params = loess_inversion(x_sel, y_sel, w, degree)

            # estimation
            filtered[ii] = loess_estimation(params, pos)

    return filtered.reshape((nx, ny))

#@nb.njit()
def loess_meshgrid(x, y):
    mx = np.empty((x.size, y.size))
    for ix in range(y.size):
        mx[:, ix] = x
    my = np.empty((x.size, y.size))
    for ix in range(x.size):
        my[ix, :] = y
    return mx, my

#@nb.njit()
def loess_distance(x, positions):
    """ compute ditances between x and positions """
    delta = positions - np.expand_dims(x, -1)
    delta_square = np.power(delta, 2)
    distance = np.sqrt(np.sum(delta_square, 0))
    return distance

#@nb.njit()
def loess_find_selection(d, l):
    """ find positions where d <= l """
    idx = np.where(d <= l)
    return idx[0]

#@nb.njit()
def loess_apply_selection(arr, idx):
    """ return array at selected positions """
    if arr.ndim == 1:
        return arr[idx]
    elif arr.ndim == 2:
        return arr[:,idx]
    else:
        raise Exception('Cannot process array with more than 2 dims')

#@nb.njit
def loess_weights(d, kernel='tricube'):
    """ compute distance based weights """
    if kernel=='tricube':
        return (1 - np.abs(d) ** 3) ** 3
    if kernel == 'gaussian':
        return np.exp(-0.5 * (d ** 2))
    if kernel == 'epanechnikov':
        return 0.75 * (1 - d  ** 2)
    raise Exception('unknown kernel %s' %kernel)

#@nb.njit()
def loess_inversion(x,y,w,degree):
    """ inversion moindres carrés """
    (ndims, nobs) = np.shape(x)
    B = np.ones((nobs, ndims*degree+1))

    idx = 1
    for ideg, deg in enumerate(range(1, degree+1)):
        for idim, dim in enumerate(range(0, ndims)):
            B[:,idx] = x[idim,:]**deg
            idx += 1

    # matrice des poids
    ww = np.diag(w)

    # inversion
    BtW = np.dot(B.T, ww)
    BtWB = np.dot(BtW, B)
    BtWBinv = pinv(BtWB)
    beta = np.dot(np.dot(BtWBinv,BtW), y.T)

    return beta

#@nb.njit()
def loess_estimation(beta, x):
    """ estimation au point courant """
    degree = int((np.shape(beta)[0]-1)/2)
    ndims = np.shape(x)[0]

    B = np.ones((1,2*degree+1))
    idx = 1
    for deg in range(1, degree+1):
        for dim in range(0, ndims):
            B[:,idx] = x[dim]**deg
            idx+=1
    return np.dot(B, beta)[0]


