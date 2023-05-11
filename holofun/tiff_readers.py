import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import warnings
from ScanImageTiffReader import ScanImageTiffReader
from tifffile import TiffFile

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
    

def get_tslice(z_idx, ch_idx, nchannels, nplanes):
    return slice((z_idx*nchannels)+ch_idx, None, nplanes*nchannels)

  
READERS = {
    'scanimage': SItiffDataReader,
    'tifffile': TiffFileDataReader
}
    

def SItiff(path: str|Path) -> AbstractReader:
    try:
        reader = READERS['scanimage'](path)
    except Exception:
        warnings.warn('Failed to load with scanimage-tiff-reader. Falling back onto tifffile...')
        reader = READERS['tifffile'](path)
    return reader