import tkinter
from tkinter import filedialog
from ScanImageTiffReader import ScanImageTiffReader
from skimage import exposure
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

MINMAX = (0,100)
CHANNEL = 'r'
CMAP = 'cividis'
# don't use these options it will fuck up your coordinates!!
XSLICE = slice(0,512)
YSLICE = slice(0,512)


def search_for_file_path():
    root = tkinter.Tk()
    root.withdraw()
    # base_dir = 'd:/frankenrig/experiments'
    # base_dir = 'f:/experiments'
    base_dir = '/mnt/hdd/data2/experiments'
    file_name = filedialog.askopenfilename(parent=root, initialdir=base_dir, 
                                           title='Select Image')
    return file_name

def load_image(file_name, channel='r', plane=0, xslice=None, yslice=None):
    if channel == 'r':
        c = 1
    elif channel == 'g':
        c = 0
    else:
        raise ValueError("Choose either 'r' or 'g'.")
    
    print('Loading file...', end=' ')
    with ScanImageTiffReader(file_name) as reader:
        data = reader.data()
    print('done.')
        
    tslice = get_tslice(file_name, plane, c)
    im_series = data[tslice,yslice,xslice]
    
    return im_series

def make_mean_image(image_series):
    im = image_series.mean(axis=0)
    im -= im.min()
    im = exposure.rescale_intensity(im, MINMAX)
    return im.squeeze()

def click_cells(im):
    plt.clf()
    plt.imshow(im, cmap=CMAP)
    plt.axis('off')
    plt.title('Click your cells. Press enter to quit.')
    pts = plt.ginput(n=-1, timeout=-1)
    plt.close()
    return np.array(pts)

def get_tslice(file:str, plane:int, channel:int) -> slice:
    """
    Determines the appropriate slicing through a ScanImageTiff depedent on how many planes,
    channels, and what plane and channel you want to slice out. Note: this returns a slice, not
    a sliced image.

    Args:
        file (str): file to slice
        plane (int): which plane you want to slice out
        channel (int): which channel you want to slice out

    Returns:
        slice: slice through time of tiff
    """
    nchannels = get_nchannels(file)
    nplanes = get_nvols(file)
    start = (plane * nchannels) + channel
    adv = nplanes * nchannels
    tslice = slice(start, -1, adv)
    return tslice
    
def get_nchannels(file):
    with ScanImageTiffReader(file) as reader:
        metadata = reader.metadata()
    channel_pass_1 = metadata.split('channelSave = [')
    if len(channel_pass_1)==1:
        nchannels = 1
    else:
        nchannels = len(metadata.split('channelSave = [')[1].split(']')[0].split(';'))
    return nchannels

def get_nvols(file):
    with ScanImageTiffReader(file) as reader:
        metadata = reader.metadata()
    if metadata.split('hStackManager.zs = ')[1][0]=='0':
        return 1
    nvols = len(metadata.split('hStackManager.zs = [')[1].split(']')[0].split(' '))
    return nvols

def make_im_rgb(image):
    rgb_im = np.zeros((*image.shape, 3))
    rgb_im[:,:,1] = image
    rgb_im[:,:,2] = image
    return rgb_im


def main():
    all_pts = []
    fpath = search_for_file_path()
    for p in range(get_nvols(fpath)):
        data = load_image(fpath, plane=p, channel=CHANNEL, xslice=XSLICE, yslice=YSLICE)
        im = make_mean_image(data)
        im = make_im_rgb(im)
        pts = click_cells(im)
        pts = [(*pt, p) for pt in pts]
        all_pts.append(pts)
    fpath = Path(fpath)
    spath = fpath.parent/'clicked_cell_locs.npy'
    all_pts = [item for sublist in all_pts for item in sublist]
    np.save(spath, all_pts)
    print(f'Saved to: {spath}')
    
if __name__ == '__main__':
    main()