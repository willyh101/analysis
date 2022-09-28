import pytest

def check_tf_missing():
    try:
        import tensorflow as tf
        return False
    except ModuleNotFoundError:
        return True

MISSING_TF = check_tf_missing()

@pytest.fixture
def tf():
    import tensorflow as tf
    return tf

@pytest.mark.skipif(MISSING_TF, reason='tensorflow not installed!')
def test_tf_version(tf):
    major_version = int(tf.__version__.split('.')[0])
    assert major_version >= 2

@pytest.mark.skipif(MISSING_TF, reason='tensorflow not installed!')    
def test_tf_cuda(tf):
    assert tf.test.is_built_with_cuda()

@pytest.mark.skipif(MISSING_TF, reason='tensorflow not installed!')    
def test_tf_gpu(tf):
    assert tf.test.is_gpu_available()
        
def test_cuda():
    try:
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cudadrv
        import atexit
        has_cuda = True
    except ImportError:
        has_cuda = False
        pytest.skip('pycuda note installed!.')
        
    assert has_cuda