import numpy as np


def series_r(trace_pA, voltage_inj_mV, start=900, stop=1200):
    amps = trace_pA[start:stop] * 10e-12
    volts = np.abs(voltage_inj_mV * 10e-3)
    return np.abs(volts/np.min(amps)) / 10e6

def input_r(trace_pA, voltage_inj_mV, start=1600, stop=1900):
    amps = trace_pA[start:stop] * 10e-12
    volts = np.abs(voltage_inj_mV * 10e-3)
    return np.abs(volts/np.mean(amps)) / 10e6

def charge_transferred(trace_pA, fs):
    return np.trapezoid(trace_pA)/fs

def charge_transferred2(trace_pA, fs):
    return np.abs(np.cumsum(trace_pA)).max()/fs