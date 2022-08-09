from ScanImageTiffReader import ScanImageTiffReader

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
    
def get_tslice(z_idx, ch_idx, nchannels, nplanes):
    return slice((z_idx*nchannels)+ch_idx, None, nplanes*nchannels)