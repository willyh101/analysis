import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from tifffile import TiffFile

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ModuleNotFoundError:
    HAS_TQDM = False

class AbstractReader(ABC):
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        
    @abstractmethod
    def read_metadata(self) -> dict:
        """Reads the metadata from the file and returns it as a dict."""
        pass
    
    @abstractmethod
    def read_data(self) -> np.ndarray:
        """Reads the data from the file and returns it as a numpy array."""
        pass
    
    def _get_nchannels(self, chans):
        if isinstance(chans, int):
            nchannels = 1
        else:
            nchannels = len(chans)
        return nchannels
    
    def _get_zs(self, zs):
        if isinstance(zs, (int, float)):
            zs = [zs]
        return zs
    
    def _get_nplanes(self, zs):
        if isinstance(zs, (int, float)):
            zs = [zs]
        return len(zs)
    
    def _get_fr(self, scanVolumeRate):
        return scanVolumeRate
    
class SItiffDataReader(AbstractReader):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.data = self.read_data()
        self._metadata = self.read_metadata()
        self.metadata = self.parse_metadata(self._metadata)
        self.reader_backend = 'scanimage'
        
    def read_data(self) -> np.ndarray:
        with ScanImageTiffReader(self.path) as reader:
            data = reader.data()
        return data
            
    def read_metadata(self):
        with ScanImageTiffReader(self.path) as reader:
            metadata = reader.metadata()
        return metadata
    
    def read_both(self):
        with ScanImageTiffReader(self.path) as reader:
            data = reader.data()
            _metadata = reader.metadata()
        return data, _metadata
    
    def parse_metadata(self, metadata) -> dict:
        """Read the SI metadata and turn in into a dict. Does not cast/eval values."""
        # split at the new line marker
        meta = metadata.split('\n')
        # filter out the ROI data by keeping fields that start with SI.
        meta = list(filter(lambda x: 'SI.' in x, meta))
        # make dictionary by splitting at equals, and only keep the last part of the fieldname as the key
        d = {k.split('.')[-1]:v for k,v in (entry.split(' = ') for entry in meta)}
        return d
    
    def _eval_numeric_metadata(self, key):
        return eval(self.metadata[key].replace(' ',',').replace(';',','))
    
    def description(self, frame):
        with ScanImageTiffReader(self.path) as reader:
            header = reader.description(frame)
        return header
    
    @property
    def nchannels(self):
        chan = self._eval_numeric_metadata('channelSave')
        return self._get_nchannels(chan)
    
    @property
    def zs(self):
        zs = self._eval_numeric_metadata('zs')
        return self._get_zs(zs)
    
    @property
    def nplanes(self):
        zs = self._eval_numeric_metadata('zs')
        return self._get_nplanes(zs)
    
    @property
    def fr(self):
        fr = self._eval_numeric_metadata('scanVolumeRate')
        return self._get_fr(fr)


class TiffFileDataReader(AbstractReader):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.data = self.read_data()
        self._metadata = self.read_metadata()
        self.metadata = self.parse_metadata(self._metadata)
        self.reader_backend = 'tifffile'
    
    def read_data(self) -> np.ndarray:
        with TiffFile(self.path) as reader:
            data = reader.asarray().copy()
        return data
    
    def read_metadata(self) -> dict:
        with TiffFile(self.path) as reader:
            metadata = reader.scanimage_metadata
        return metadata
    
    def parse_metadata(self, metadata) -> dict:
        meta = {k:v for k,v in metadata['FrameData'].items() if k.startswith('SI.')}
        d = {k.split('.')[-1]:v for k,v in meta.items()}
        return d
    
    @property
    def nchannels(self):
        chan = self.metadata['channelSave']
        return self._get_nchannels(chan)
    
    @property
    def zs(self):
        zs = self.metadata['zs']
        return self._get_zs(zs)
    
    @property
    def nplanes(self):
        zs = self.metadata['zs']
        return self._get_nplanes(zs)
    
    @property
    def fr(self):
        fr = self.metadata['scanVolumeRate']
        return self._get_fr(fr)
    

def get_tslice(z_idx: int, ch_idx: int, nchannels: int, nplanes: int):
    """
    Get the slice of the data corresponding to a specified zplane and channel.

    Args:
        z_idx (int): z-plane index
        ch_idx (int): channel index
        nchannels (int): total number of channels
        nplanes (int): total number of z-planes

    Returns:
        slice: slicer into the data array
    """
    return slice((z_idx*nchannels)+ch_idx, None, nplanes*nchannels)

READERS = {
    'scanimage': SItiffDataReader,
    'tifffile': TiffFileDataReader
}

class SItiffCore:
    def __init__(self, path: Path, backend: str = 'scanimage') -> None:
        self.path = str(path)
        
        self.backend = self._get_reader(backend)
        
        try:
            self._reader = self.backend(self.path)
        except Exception:
            warnings.warn(f'Could not read {self.path} with {self.backend}. Trying TiffFileDataReader.')
            self.backend = TiffFileDataReader
            self._reader = self.backend(self.path)
        
        self.data = self._reader.data
        self._metadata = self._reader._metadata
        self.metadata = self._reader.metadata
        
        self.nchannels = self._reader.nchannels
        self.zs = self._reader.zs
        self.nplanes = self._reader.nplanes
        self.fr = self._reader.fr
        
    def _get_reader(self, backend: str) -> AbstractReader:
        try:
            return READERS[backend]
        except KeyError:
            raise ValueError(f'Could not find backend {backend}. Available options are {list(READERS.keys())}')
        
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
    data = SItiffCore(mov_path).data
    data = data[t_slice, y_slice, x_slice]
    return data
    
def tiffs2array(movie_list, x_slice, y_slice, t_slice):
    data = [slice_movie(str(mov), x_slice, y_slice, t_slice) for mov in movie_list]
    return np.concatenate(data)

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