import numpy as np
from holofun.daq import calculate_trial_run_speed

def test_calculate_trial_run_speed():
    # Test case 1: Basic test case with default parameters
    rotary_sweep = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    fr = 10.0
    expected_speed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(calculate_trial_run_speed(rotary_sweep, fr), expected_speed)

    # Test case 2: Custom parameters
    rotary_sweep = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    fr = 5.0
    daq_rate = 10000
    wheel_circum = 50.0
    expected_speed = np.array([0.0, 0.0, 0.0, 0.0])
    assert np.allclose(calculate_trial_run_speed(rotary_sweep, fr, daq_rate, wheel_circum), expected_speed)

    # Test case 3: Empty input
    rotary_sweep = np.array([])
    fr = 30.0
    expected_speed = np.array([])
    assert np.allclose(calculate_trial_run_speed(rotary_sweep, fr), expected_speed)

    # Add more test cases as needed