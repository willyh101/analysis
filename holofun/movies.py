import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from ScanImageTiffReader import ScanImageTiffReader
from tqdm import tqdm
import time
from sklearn.decomposition import PCA


from .tiffs import SItiff, get_tslice


def make_mov_array(tiff_list, zplane, ch=0, x_cut=slice(None), y_cut=slice(None)):
    """
    Make a frame by YX movie from tiff list.

    Args:
        tiff_list (list): list of files to use
        zplane (int): idx of zplane to use
        ch (int, optional): channel to make a movie from. Defaults to 0.
        x_cut (slice, optional): slice range to cut. Defaults to slice(None).
        y_cut (slice, optional): slice range to cut. Defaults to slice(None).

    Returns:
        np.array of frames in concatenated movie
    """
    
    first_tiff = SItiff(tiff_list[0])
    t_slice = get_tslice(zplane, ch, first_tiff.nchannels, first_tiff.nplanes)
    
    lengths = []
    mov_array = []
    
    for tiff in tqdm(tiff_list, desc='Loading tiffs: '):
        with ScanImageTiffReader(str(tiff)) as reader:
            data = reader.data()
            data = data[t_slice, y_cut, x_cut]
            mov_array.append(data)
            lengths.append(data.shape[0])
            
    mov_array = np.concatenate(mov_array)
    lengths = np.array(lengths)
    
    return mov_array, lengths

def make_mean_movie(mov_array, tiff_lengths):
    mov_cut = np.split(mov_array, np.cumsum(tiff_lengths[:-1]), axis=0)
    shortest = min(map(lambda x: x.shape[0], mov_cut))
    mmov = np.array([a[:shortest,:,:] for a in mov_cut]).mean(0)
    return mmov

def play_movie(mov, fps, vmin=None, vmax=None, cmap='viridis', figsize=(1,1), text_start_stop=None, 
               text=None, text_color='k', fontsize=2, ax=None):
    update_ms = int((1/fps)*1000)
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax: plt.Axes

    plt.close() 
    
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow(mov[0,:,:], cmap=cmap, vmin=vmin, vmax=vmax)

    def animate(i):
        img.set_array(mov[i,:,:])
        if text_start_stop is not None:
            if i in range(*text_start_stop):
                ax.text(0.5, 0.9, text, fontsize=fontsize, horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, color=text_color)
            else:
                for t in ax.texts:
                    t.remove()

    ani = animation.FuncAnimation(fig, animate, frames=mov.shape[0], interval=update_ms)
    
    return ani

BLOCK_SIZE = np.array([128, 128])/2
N_COMPS_FRAC = 0.5

def pca_denoise(mov: np.ndarray, block_size=BLOCK_SIZE, n_comps_frac=N_COMPS_FRAC):
    t0 = time.time()
    nframes, Ly, Lx = mov.shape
    yblock, xblock, _, block_size, _ = make_blocks(Ly, Lx, block_size=block_size)

    mov_mean = mov.mean(axis=0)
    mov -= mov_mean

    nblocks = len(yblock)
    Lyb, Lxb = block_size
    n_comps = int(min(min(Lyb * Lxb, nframes), min(Lyb, Lxb) * n_comps_frac))
    maskMul = spatial_taper(Lyb // 4, Lyb, Lxb)
    norm = np.zeros((Ly, Lx), np.float32)
    reconstruction = np.zeros_like(mov)
    block_re = np.zeros((nblocks, nframes, Lyb * Lxb))
    for i in range(nblocks):
        block = mov[:, yblock[i][0]:yblock[i][-1],
                    xblock[i][0]:xblock[i][-1]].reshape(-1, Lyb * Lxb)
        model = PCA(n_components=n_comps, random_state=0).fit(block)
        block_re[i] = (block @ model.components_.T) @ model.components_
        norm[yblock[i][0]:yblock[i][-1], xblock[i][0]:xblock[i][-1]] += maskMul

    block_re = block_re.reshape(nblocks, nframes, Lyb, Lxb)
    block_re *= maskMul
    for i in range(nblocks):
        reconstruction[:, yblock[i][0]:yblock[i][-1],
                       xblock[i][0]:xblock[i][-1]] += block_re[i]
    reconstruction /= norm
    print("Binned movie denoised in %0.2f sec." %
          (time.time() - t0))
    reconstruction += mov_mean
    return reconstruction


def make_blocks(Ly, Lx, block_size=(128, 128)):
    """
    Computes overlapping blocks to split FOV into to register separately

    Parameters
    ----------
    Ly: int
        Number of pixels in the vertical dimension
    Lx: int
        Number of pixels in the horizontal dimension
    block_size: int, int
        block size

    Returns
    -------
    yblock: float array
    xblock: float array
    nblocks: int, int
    block_size: int, int
    NRsm: array
    """

    block_size_y, ny = calculate_nblocks(L=Ly, block_size=block_size[0])
    block_size_x, nx = calculate_nblocks(L=Lx, block_size=block_size[1])
    block_size = (block_size_y, block_size_x)

    # todo: could rounding to int here over-represent some pixels over others?
    ystart = np.linspace(0, Ly - block_size[0], ny).astype("int")
    xstart = np.linspace(0, Lx - block_size[1], nx).astype("int")
    yblock = [
        np.array([ystart[iy], ystart[iy] + block_size[0]])
        for iy in range(ny)
        for _ in range(nx)
    ]
    xblock = [
        np.array([xstart[ix], xstart[ix] + block_size[1]])
        for _ in range(ny)
        for ix in range(nx)
    ]

    NRsm = kernelD2(xs=np.arange(nx), ys=np.arange(ny)).T

    return yblock, xblock, [ny, nx], block_size, NRsm

def calculate_nblocks(L: int, block_size: int = 128) -> tuple[int, int]:
    """
    Returns block_size and nblocks from dimension length and desired block size

    Parameters
    ----------
    L: int
    block_size: int

    Returns
    -------
    block_size: int
    nblocks: int
    """
    return (L, 1) if block_size >= L else (block_size,
                                           int(np.ceil(1.5 * L / block_size)))

def kernelD2(xs: int, ys: int) -> np.ndarray:
    ys, xs = np.meshgrid(xs, ys)
    ys = ys.flatten().reshape(1, -1)
    xs = xs.flatten().reshape(1, -1)
    R = np.exp(-((ys - ys.T)**2 + (xs - xs.T)**2))
    R = R / np.sum(R, axis=0)
    return R

def spatial_taper(sig, Ly, Lx):
    """
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


    """
    xx, yy = meshgrid_mean_centered(x=Lx, y=Ly)
    mY = ((Ly - 1) / 2) - 2 * sig
    mX = ((Lx - 1) / 2) - 2 * sig
    maskY = 1. / (1. + np.exp((yy - mY) / sig))
    maskX = 1. / (1. + np.exp((xx - mX) / sig))
    maskMul = maskY * maskX
    return maskMul

def meshgrid_mean_centered(x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
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