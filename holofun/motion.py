import numpy as np
from .motion_utils import *

from .utils import tic, ptoc

SMOOTH_SIGMA = 1
MAX_SHIFT = 10
NORM = True

def pick_initial_reference(frames: np.ndarray):
    """ computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    nimg,Ly,Lx = frames.shape
    frames = np.reshape(frames, (nimg,-1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = np.matmul(frames, frames.T)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis = 1)
    bestCC = np.mean(CCsort[:, 1:20], axis=1)
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    refImg = np.mean(frames[indsort[0:20], :], axis = 0)
    refImg = np.reshape(refImg, (Ly,Lx))
    return refImg

def compute_reference_img(frames:np.ndarray, refImg:np.ndarray):
    t1 = tic()
    niter = 5
    for i in range(niter):
        print(f'reference alignment round #{i}')
        t2 = tic()
        masks = compute_masks(refImg, maskSlope=SMOOTH_SIGMA*3)
        data = apply_masks(frames, *masks)
        cfRefImg = phasecorr_reference(refImg, smooth_sigma=SMOOTH_SIGMA)
        ymax, xmax, cmax = phasecorr(data, cfRefImg, maxregshift=MAX_SHIFT, smooth_sigma_time=0)
        for frame, dy, dx in zip(frames, ymax, xmax):
            frame[:] = shift_frame(frame, dy, dx)
        nmax = max(2, int(frames.shape[0] * (1 + i) / (2 * niter)))
        isort = np.argsort(-cmax)[1:nmax]
        refImg = frames[isort].mean(axis=0).astype(np.int16)
        refImg = shift_frame(
            frame=refImg,
            dx=int(np.round(-ymax[isort].mean())),
            dy=int(np.round(-xmax[isort].mean()))
        )
        print(f'xmax: {xmax.max()}, ymax: {ymax.max()}')
        ptoc(t2, 'Round finished in:')
    ptoc(t1, 'Reference image found.')
    return refImg

def register_mov(mov, refImg, batch_size=500):
    rmin, rmax = np.int16(np.percentile(refImg,5)), np.int16(np.percentile(refImg,99.5))
    maskMul, maskOffset = compute_masks(refImg, maskSlope=SMOOTH_SIGMA*3)
    cfRefImg = phasecorr_reference(refImg, smooth_sigma=SMOOTH_SIGMA)
    n_frames, Ly, Lx = mov.shape
    rigid_offsets = []
    for k in np.arange(0, n_frames, batch_size):
        if k // 1:
            print('starting batch', k)
        frames = mov[k : min(k + batch_size, n_frames)]
        data = apply_masks(np.clip(frames, rmin, rmax), maskMul,maskOffset)
        ymax, xmax, cmax = phasecorr(data, cfRefImg, MAX_SHIFT, smooth_sigma_time=0)
        for frame, dy, dx in zip(frames, ymax, xmax):
            frame[:] = shift_frame(frame, dy, dx)
        rigid_offsets.append([ymax, xmax, cmax])
    rigid_offsets = combine_offsets_across_batches(rigid_offsets, rigid=True)
    return mov, rigid_offsets

def run_registration(mov, n_init=500, batch_size=1000):
    frames = mov[np.linspace(0, mov.shape[0], 1+n_init, dtype=int)[:-1]]
    ref_frames = pick_initial_reference(frames)
    ref = compute_reference_img(frames, ref_frames)
    mov, offsets = register_mov(mov, ref, batch_size)
    return mov, offsets