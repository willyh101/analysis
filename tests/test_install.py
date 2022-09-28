import pytest

def test_mkl_install():
    import numpy as np
    assert np.__config__.get_info('blas_mkl_info'), 'No MKL found!'