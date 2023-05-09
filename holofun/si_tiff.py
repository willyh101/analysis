import numpy as np
from pathlib import Path
import pickle
from ScanImageTiffReader import ScanImageTiffReader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ModuleNotFoundError:
    HAS_TQDM = False

class SItiffCore:
    def __init__(self, path) -> None:
        """
        Loads a ScanImage tiff from the specified path.

        Args:
            path (str or Path): filepath to tiff
        """
        self.path = str(path)
        
        with ScanImageTiffReader(self.path) as reader:
            self.data = reader.data()
            self._metadata = reader.metadata()
        
        self.metadata = metadata_to_dict(self._metadata)
            
    def _eval_numeric_metadata(self, key):
        return eval(self.metadata[key].replace(' ',',').replace(';',','))
    
    @property
    def nchannels(self):
        chans = self._eval_numeric_metadata('channelSave')
        if isinstance(chans, int):
            nchannels = 1
        else:
            nchannels = len(chans)
        return nchannels
    
    @property
    def zs(self):
        zs = self._eval_numeric_metadata('zs')
        if isinstance(zs, (int, float)):
            zs = [zs]
        return zs
    
    @property
    def nplanes(self):
        return len(self.zs)
    
    @property
    def fr(self):
        return self._eval_numeric_metadata('scanVolumeRate')
    
    def description(self, frame):
        with ScanImageTiffReader(self.path) as reader:
            header = reader.description(frame)
        return header
    
    def extract(self, z, ch):
        """
        Get the underlying data for a specified zplane and channel. 

        Args:
            z (int): Index of zplane to extract.
            ch (int): Index of channel to extract. Use 0 for channel 1/green PMT, 1 for channel 2/red PMT.
            
        Returns:
            nframes x N x N numpy array slice of data
        """
        tslice = get_tslice(z, ch, self.nchannels, self.nplanes)
        img = self.data[tslice,:,:]
        return img
    
def load_si_metadata_as_dict(file) -> dict:
    """Read the SI metadata and turn in into a dict. Does not cast/eval values."""
    with ScanImageTiffReader(file) as reader:
        meta = reader.metadata()
    d = metadata_to_dict(meta)
    return d

def metadata_to_dict(meta) -> dict:
    """Read the SI metadata and turn in into a dict. Does not cast/eval values."""
    # split at the new line marker
    meta = meta.split('\n')
    # filter out the ROI data by keeping fields that start with SI.
    meta = list(filter(lambda x: 'SI.' in x, meta))
    # make dictionary by splitting at equals, and only keep the last part of the fieldname as the key
    d = {k.split('.')[-1]:v for k,v in (entry.split(' = ') for entry in meta)}
    return d
    


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
        try_load (bool): Attempts to load the pickle file with list of tiff lengths

    Returns:
        numpy array of tiff/trial lengths
    """
    
    first_tiff = SItiffCore(movie_list[0])
    t_slice = get_tslice(0, 0, first_tiff.nchannels, first_tiff.nplanes)
    if HAS_TQDM:
        nframes = [slice_movie(str(mov), slice(None), slice(None), t_slice).shape[0] for mov in tqdm(movie_list, desc="Counting Tiffs: ")]
    else:
        nframes = [slice_movie(str(mov), slice(None), slice(None), t_slice).shape[0] for mov in movie_list]
    
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

def get_tslice(z_idx, ch_idx, nchannels, nplanes):
    return slice((z_idx*nchannels)+ch_idx, None, nplanes*nchannels)