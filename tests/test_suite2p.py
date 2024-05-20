from pathlib import Path
import pytest

from holofun.s2p import Suite2pData

@pytest.fixture
def test_folder() -> str:
    return '/mnt/nvme/data/suite2p_outputs/w75_1/20240509/1stim_2ret_3ori_4oriSmall_6expt/suite2p'

@pytest.fixture
def s2p(test_folder) -> Suite2pData:
    return Suite2pData(test_folder)

def test_s2p_init(test_folder: str):
    assert Path(test_folder).exists()
    tmp_s2p = Suite2pData(test_folder)
    assert isinstance(tmp_s2p, Suite2pData)
    
def test_planes_in_order(s2p: Suite2pData):
    plane_idxs = [int(p.stem[-1]) for p in s2p.plane_folders]
    assert plane_idxs == sorted(plane_idxs)