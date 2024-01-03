from typing import Tuple

import numpy as np
from mkl_fft import fft2, ifft2
from numba import complex64, vectorize
from numpy.fft import ifftshift
from scipy.fft import next_fast_len
from scipy.ndimage import gaussian_filter1d


def convolve(mov: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Returns the 3D array 'mov' convolved by a 2D array 'img'.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to process
    img: 2D array
        The convolution kernel

    Returns
    -------
    convolved_data: nImg x Ly x Lx
    """
    return ifft2(apply_dotnorm(fft2(mov), img))

@vectorize([complex64(complex64, complex64)], nopython=True, target='parallel')
def apply_dotnorm(Y, cfRefImg):
    return Y / (np.complex64(1e-5) + np.abs(Y)) * cfRefImg

@vectorize(['complex64(int16, float32, float32)', 'complex64(float32, float32, float32)'], nopython=True, target='parallel', cache=True)
def addmultiply(x, mul, add):
    return np.complex64(np.float32(x) * mul + add)

def spatial_taper(sig, Ly, Lx):
    '''
    Returns spatial taper  on edges with gaussian of std sig

    Parameters
    ----------
    sig
    Ly: int
        frame height
    Lx: int
        frame width

    Returns
    -------
    maskMul


    '''
    xx, yy = meshgrid_mean_centered(x=Lx, y=Ly)
    mY = ((Ly - 1) / 2) - 2 * sig
    mX = ((Lx - 1) / 2) - 2 * sig
    maskY = 1. / (1. + np.exp((yy - mY) / sig))
    maskX = 1. / (1. + np.exp((xx - mX) / sig))
    maskMul = maskY * maskX
    return maskMul

def meshgrid_mean_centered(x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a mean-centered meshgrid

    Parameters
    ----------
    x: int
        The height of the meshgrid
    y: int
        The width of the mehgrid

    Returns
    -------
    xx: int array
    yy: int array
    """
    x = np.arange(0, x)
    y = np.arange(0, y)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def temporal_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Returns Gaussian filtered 'frames' ndarray over first dimension

    Parameters
    ----------
    data: nImg x Ly x Lx
    sigma: float
        windowing parameter

    Returns
    -------
    smoothed_data: nImg x Ly x Lx
        Smoothed data

    """
    return gaussian_filter1d(data, sigma=sigma, axis=0)

def phasecorr_reference(refImg: np.ndarray, smooth_sigma=None) -> np.ndarray:
    """
    Returns reference image fft'ed and complex conjugate and multiplied by gaussian filter in the fft domain,
    with standard deviation 'smooth_sigma' computes fft'ed reference image for phasecorr.

    Parameters
    ----------
    refImg : 2D array, int16
        reference image

    Returns
    -------
    cfRefImg : 2D array, complex64
    """
    cfRefImg = complex_fft2(img=refImg)
    cfRefImg /= (1e-5 + np.absolute(cfRefImg))
    cfRefImg *= gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    return cfRefImg.astype('complex64')

def complex_fft2(img: np.ndarray, pad_fft: bool = False) -> np.ndarray:
    """
    Returns the complex conjugate of the fft-transformed 2D array 'img', optionally padded for speed.

    Parameters
    ----------
    img: Ly x Lx
        The image to process
    pad_fft: bool
        Whether to pad the image


    """
    Ly, Lx = img.shape
    return np.conj(fft2(img, (next_fast_len(Ly), next_fast_len(Lx)))) if pad_fft else np.conj(fft2(img))

def gaussian_fft(sig, Ly: int, Lx: int):
    '''
    gaussian filter in the fft domain with std sig and size Ly,Lx

    Parameters
    ----------
    sig
    Ly: int
        frame height
    Lx: int
        frame width

    Returns
    -------
    fhg: np.ndarray
        smoothing filter in Fourier domain

    '''
    xx, yy = meshgrid_mean_centered(x=Lx, y=Ly)
    hgx = np.exp(-np.square(xx/sig) / 2)
    hgy = np.exp(-np.square(yy/sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft2(ifftshift(hgg)))
    return fhg

def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Returns frame, shifted by dy and dx

    Parameters
    ----------
    frame: Ly x Lx
    dy: int
        vertical shift amount
    dx: int
        horizontal shift amount

    Returns
    -------
    frame_shifted: Ly x Lx
        The shifted frame

    """
    return np.roll(frame, (-dy, -dx), axis=(0, 1))

def combine_offsets_across_batches(offset_list, rigid):
    yoff, xoff, corr_xy = [], [], []
    for batch in offset_list:
        yoff.append(batch[0])
        xoff.append(batch[1])
        corr_xy.append(batch[2])
    if rigid:
        return np.hstack(yoff), np.hstack(xoff), np.hstack(corr_xy)
    else:
        return np.vstack(yoff), np.vstack(xoff), np.vstack(corr_xy)
    
def compute_masks(refImg, maskSlope) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns maskMul and maskOffset from an image and slope parameter

    Parameters
    ----------
    refImg: Ly x Lx
        The image
    maskSlope

    Returns
    -------
    maskMul: float arrray
    maskOffset: float array
    """
    Ly, Lx = refImg.shape
    maskMul = spatial_taper(maskSlope, Ly, Lx)
    maskOffset = refImg.mean() * (1. - maskMul)
    return maskMul.astype('float32'), maskOffset.astype('float32')

def apply_masks(data: np.ndarray, maskMul: np.ndarray, maskOffset: np.ndarray) -> np.ndarray:
    """
    Returns a 3D image 'data', multiplied by 'maskMul' and then added 'maskOffet'.

    Parameters
    ----------
    data: nImg x Ly x Lx
    maskMul
    maskOffset

    Returns
    --------
    maskedData: nImg x Ly x Lx
    """
    return addmultiply(data, maskMul, maskOffset)

def phasecorr(data, cfRefImg, maxregshift, smooth_sigma_time) -> Tuple[int, int, float]:
    """ compute phase correlation between data and reference image

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    maxregshift : float
        maximum shift as a fraction of the minimum dimension of data (min(Ly,Lx) * maxregshift)
    smooth_sigma_time : float
        how many frames to smooth in time

    Returns
    -------
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame
    cmax : float
        maximum of phase correlation for each frame

    """
    min_dim = np.minimum(*data.shape[1:])  # maximum registration shift allowed
    lcorr = int(np.minimum(np.round(maxregshift * min_dim), min_dim // 2))
    
    #cc = convolve(data, cfRefImg, lcorr)
    data = convolve(data, cfRefImg)
    cc = np.real(np.block(
                [[data[:,  -lcorr:, -lcorr:], data[:,  -lcorr:, :lcorr+1]],
                [data[:, :lcorr+1, -lcorr:], data[:, :lcorr+1, :lcorr+1]]]
            )
            )
        
    cc = temporal_smooth(cc, smooth_sigma_time) if smooth_sigma_time > 0 else cc

    ymax, xmax = np.zeros(data.shape[0], np.int32), np.zeros(data.shape[0], np.int32)
    for t in np.arange(data.shape[0]):
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1))
    cmax = cc[np.arange(len(cc)), ymax, xmax]
    ymax, xmax = ymax - lcorr, xmax - lcorr

    return ymax, xmax, cmax.astype(np.float32)

def phasecorr_reference(refImg: np.ndarray, smooth_sigma=None) -> np.ndarray:
    """
    Returns reference image fft'ed and complex conjugate and multiplied by gaussian filter in the fft domain,
    with standard deviation 'smooth_sigma' computes fft'ed reference image for phasecorr.

    Parameters
    ----------
    refImg : 2D array, int16
        reference image

    Returns
    -------
    cfRefImg : 2D array, complex64
    """
    cfRefImg = complex_fft2(img=refImg)
    cfRefImg /= (1e-5 + np.absolute(cfRefImg))
    cfRefImg *= gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    return cfRefImg.astype('complex64')