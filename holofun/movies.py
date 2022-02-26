from ScanImageTiffReader import ScanImageTiffReader
from .tiffs import SItiff, get_tslice
import numpy as np
from tqdm import tqdm
from matplotlib import animation
import matplotlib.pyplot as plt


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

def play_movie(mov, fps):
    update_ms = int((1/fps)*1000)
    
    fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=300)
    plt.close()
    
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow(mov[0,:,:], cmap='viridis')

    def animate(i):
        img.set_data(mov[i,:,:])

    ani = animation.FuncAnimation(fig, animate, frames=mov.shape[0], interval=update_ms)
    
    return ani