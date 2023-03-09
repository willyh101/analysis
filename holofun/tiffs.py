import pickle
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from ScanImageTiffReader import ScanImageTiffReader
from skimage import exposure
from tqdm import tqdm

from .constants import PX_PER_UM
from .si_tiff import SItiffCore, get_tslice
from .simple_guis import openfilegui

class SItiff(SItiffCore):
        
    def mean_img(self, z_idx, channel, scaling=None, as_rgb=False, rgb_ch=None, blue_as_cyan=True):
        """
        Get the mean image of a specified z-plane and channel. Takes the mean and does a minimum
        subtract. Optionally rescales the LUT intensity. Optionally returns as RGB-compatible array.

        Args:
            z_idx (int): Index of zplane to extract.
            channel (int): Index of channel to extract. Use 0 for channel 1/green PMT, 1 for channel 2/red PMT.
            scaling (tuple of ints, optional): LUT range to use. Defaults to None for no scaling.
            as_rgb (bool, optional): Whether to return a NxNx3 RGB image. Defaults to False.
            rgb_ch (int, optional): Specifies the RGB channel to use. Defaults to None.
            blue_as_cyan (bool, optional): [description]. Defaults to True.

        Returns:
            NxN numpy array of image (or NxNx3 if RGB)
        """
        tslice = get_tslice(z_idx, channel, self.nchannels, self.nplanes)
        mimg = np.mean(self.data[tslice,:,:], axis=0)
        mimg -= mimg.min()
        
        if scaling:
            mimg = exposure.rescale_intensity(mimg, scaling)
            
        if as_rgb:
            mimg = exposure.rescale_intensity(mimg, out_range=np.uint8)
            rgb_im = np.zeros((*mimg.shape,3), dtype=np.uint8)
            if not rgb_ch:
                if channel == 1:
                    rgb_ch = 0
                elif channel == 0:
                    rgb_ch = 1
                else: 
                    rgb_ch = channel
            rgb_im[:,:,rgb_ch] = mimg
            
            if blue_as_cyan and rgb_ch == 2:
                rgb_im[:,:,1] = mimg
            
            return rgb_im
        
        else:   
            return mimg
    
    def merge_mean(self, z_idx, gscale=None, rscale=None, green_as_cyan=False):
        green_ch = self.mean_img(z_idx, 0, scaling=gscale)
        red_ch = self.mean_img(z_idx, 1, scaling=rscale)
        
        rgb_im = np.zeros((*green_ch.shape, 3))
        rgb_im[:,:,0] = red_ch
        rgb_im[:,:,1] = green_ch
        
        if green_as_cyan:
            rgb_im[:,:,2] = green_ch
            
        return rgb_im
    
    def show(self, z_idx, ch_idx, scaling=None, as_rgb=False, rgb_ch=None, ax=None, 
             ch_label_txt=None, ch_label_color='white', **kwargs):
        
        if ax is None:
            ax = plt.gca()
            
        mean_img = self.mean_img(z_idx, ch_idx, scaling=scaling, as_rgb=as_rgb, rgb_ch=rgb_ch, **kwargs)
        
        
        # note I don't think this will work will multiple colors since you can't specify per line
        fontdict = {
            'size':18,
            'color':ch_label_color
        }
        label = AnchoredText(ch_label_txt, loc='upper left', prop=fontdict,
                             frameon=False, pad=0.2, borderpad=0.2)
        
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(ax.transData, 
                                128,
                                f'200 $\mu$m',
                                'lower right',
                                pad=0.2,
                                color='white',
                                frameon=False,
                                size_vertical=8,
                                fontproperties=fontprops)
        ax.add_artist(scalebar)
        ax.add_artist(label)
        ax.imshow(mean_img)
        ax.axis('off')
        
        return ax
    
    @classmethod
    def load(cls, rootdir='d:/frankenrig/experiments'):
        path = openfilegui(rootdir=rootdir, title='Select Tiff')
        if not path:
            return
        print(f'Loading tiff: {path}')
        return cls(path)

def slice_movie(mov_path, x_slice, y_slice, t_slice) -> np.ndarray:
    """
    Slice a single tiff along x, y, and time dims. Time dim must account for number of channels and
    z-planes. slice((z_idx*nchannels)+channel, -1, nplanes*nchannels)

    Args:
        mov_path (str): path to movie
        x_slice (slice): slice along x-axis
        y_slice (slice): slice along y-axis
        t_slice (slice): slice along t-axis

    Returns:
        np.array: array of sliced movie
    """
    with ScanImageTiffReader(mov_path) as reader:
        data = reader.data()
        data = data[t_slice, y_slice, x_slice]
    return data

def count_tiff_lengths(movie_list, save=False):
    """
    Counts the length of tiffs for a single plane and channel to get trial lenths. Optionally save
    the data to dist as a pickle.

    Args:
        movie_list (list): list of str or Path pointing to the tiffs to count.
        save (bool or str, optional): Whether to save the file. can either set to True to save in
                                      in the folder with the counted tiffs or specify save location
                                      with a string. Defaults to False.

    Returns:
        numpy array of tiff/trial lengths
    """
    
    first_tiff = SItiff(movie_list[0])
    t_slice = get_tslice(0, 0, first_tiff.nchannels, first_tiff.nplanes)
    nframes = [slice_movie(str(mov), slice(None), slice(None), t_slice).shape[0] for mov in tqdm(movie_list, desc="Counting Tiffs: ")]
    
    if save:
        if isinstance(save, str):
            save_path = Path(save, 'tiff_lengths.pickle')
        else:
            save_path = Path(first_tiff.path).parent/'tiff_lengths.pickle'
            
        with open(save_path, 'wb') as f:
            pickle.dump(nframes, f)
    
    return np.array(nframes)

def tiffs2array(movie_list, x_slice, y_slice, t_slice):
    data = [slice_movie(str(mov), x_slice, y_slice, t_slice) for mov in movie_list]
    return np.concatenate(data)

def get_crop_mask(Cx, Cy, bb):
    """
    Returns a mask to crop an image around a centered x,y point. Uses advanced indexing.

    Args:
        Cx (int, float): center X location
        Cy (int, float): center Y location
        bb (int): bounding box on either side of Cx, Cy
    """
    xs = np.arange(Cx-bb, Cx+bb, dtype=int)
    ys = np.arange(Cy-bb, Cy+bb, dtype=int)
    mask = np.ix_(ys,xs)
    return mask

def create_rgb_img(*args):
    for arg in args:
        if arg is not None:
            rgb_im = np.zeros((*arg.shape, 3), dtype=np.uint8)
    for i,img in enumerate(args):
        if img is None:
            continue
        else:
            img -= img.min()
            new_img = exposure.rescale_intensity(img, out_range=np.uint8)
            rgb_im[:,:,i] = new_img
    return rgb_im

class RGBImgViewer:
    def __init__(self, *args):
        if len(args) > 1 and len(args) < 4:
            self.raw_data = create_rgb_img(*args)
        elif len(args) == 1:
            if len(args[0].shape) == 3:
                assert args[0].shape[-1] == 3, 'Data must be (m,n,3) for single argument ndarry.'
                self.raw_data = args[0]
            else:
                self.raw_data
            self.raw_data = args[0]
        self.img = self.raw_data.copy()
    
    def rescale(self, ch, upper=255, lower=0):
        ch_data = self.raw_data[:,:,ch]
        ch_data_adj = exposure.rescale_intensity(ch_data, (lower, upper))
        self.img[:,:,ch] = ch_data_adj
        
    def reset(self):
        self.img = self.raw_data.copy()
        
    def show(self, ch=None, ax=None):
        if ax is None:
            ax = plt.gca()
        
        if ch is None:
            rgb = self.img
        else:
            rgb = np.zeros_like(self.img)
            rgb[:,:,ch] = self.img[:,:,ch]
        
        ax.imshow(rgb)
        ax.axis('off')
        
def add_scalebar(ax, um_length, fs=18, lw=8, **kwargs):
    px_length = um_length * PX_PER_UM
    fontprops = fm.FontProperties(size=fs)
    scalebar = AnchoredSizeBar(ax.transData, 
                            px_length,
                            f'{um_length} µm',
                            'lower right',
                            pad=0.2,
                            borderpad=0.2,
                            color='white',
                            frameon=False,
                            size_vertical=lw,
                            fontproperties=fontprops, **kwargs)
    ax.add_artist(scalebar)
    return ax

def add_label(ax, txt, c='white', sz=12, font_kw=None, pad=0.3, **kwargs):
    font_dict = {
        'size':sz,
        'color': c
    }
    if font_kw is not None:
        font_dict = {**font_kw, **font_dict}
        
    kwargs.setdefault('pad',pad)
    kwargs.setdefault('borderpad', pad)
    kwargs.setdefault('loc', 'upper left')
    label = AnchoredText(txt, prop=font_dict, frameon=False, **kwargs)
    ax.add_artist(label)
    return ax