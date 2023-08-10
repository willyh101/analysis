import numpy as np
import pytest
from holofun.si_tiff import SItiffCore, TiffFileDataReader, SItiffDataReader

@pytest.fixture
def good_file():
    return '/mnt/data2/experiments/w57_1/20230125/20230125_w57_1_img1020_00001.tif'

@pytest.fixture
def test_file_fails():
    return '/mnt/scratch/masato/FOV1.tif'

@pytest.fixture(params=[SItiffDataReader, TiffFileDataReader])
def reader_backend(request):
    return request.param

@pytest.fixture
def tif(good_file, reader_backend):
    return SItiffCore(good_file, backend=reader_backend)

NPLANES = 3
NCHANNELS = 2

# pytestmark = pytest.mark.parametrize('backend', [SItiffDataReader, TiffFileDataReader])

def test_data(tif):
    assert isinstance(tif.data, np.ndarray)
    
def test_metadata(tif):
    assert isinstance(tif.metadata, dict)
    
def test_extract(tif):
    tmp = tif.extract(0, 0)
    assert isinstance(tmp, np.ndarray)
    
def test_nplanes(tif):
    assert tif.nplanes == NPLANES
    assert isinstance(tif.nplanes, int)
    
def test_nchannels(tif):
    assert tif.nchannels == NCHANNELS
    assert isinstance(tif.nchannels, int)
    
def test_zs(tif):
    assert isinstance(tif.zs, list)
    assert len(tif.zs) == NPLANES
    assert isinstance(tif.zs[0], (float, int))
    
def test_backend_failover_to_tifffile(test_file_fails):
    tif = SItiffCore(test_file_fails, backend=SItiffDataReader)
    assert isinstance(tif._reader, TiffFileDataReader)
    assert tif._reader.reader_backend == 'tifffile'