import numpy as np


def series_r(trace_nA, voltage_inj_mV, start=900, stop=1200):
    trace = trace_nA[start:stop]
    volts = np.abs(voltage_inj_mV * 10e-3)
    amps = trace * 10e-9
    return np.abs(volts/np.min(amps)) / 10e6

def input_r(trace_nA, voltage_inj_mV, start=1500, stop=1900):
    trace = trace_nA[start:stop]
    volts = np.abs(voltage_inj_mV * 10e-3)
    amps = trace * 10e-9
    return np.abs(volts/np.mean(amps)) / 10e6

def charge_transferred(trace_nA):
    return np.trapezoid(trace_nA)

def charge_transferred2(trace_nA):
    return np.abs(np.cumsum(trace_nA)).max()