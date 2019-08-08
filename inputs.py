from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy


def _circular_track_firing_rates(track_length=1, dt=0.1, loops=2, speed=1, amplitude= 30, variance = .2, neuron_groups=10):
    '''
    '''
    x = 2*np.pi/track_length*np.linspace(0, loops*track_length, int(loops*track_length/speed/dt+1))
    y_locations = np.sin(x)
    x_locations = np.cos(x)

    locations = np.arctan2(y_locations, x_locations)
    nrn_locations = (np.linspace(-np.pi,np.pi, neuron_groups))

    N_nrn = nrn_locations.shape[0]
    N_t = locations.shape[0]
    locations_norm = abs(np.tile(locations, [N_nrn, 1]) - np.tile(nrn_locations,[N_t,1]).T)
    locations_norm[locations_norm>np.pi] = 2*np.pi-locations_norm[abs(locations_norm)>np.pi]

    rates = amplitude*np.exp(-(locations_norm**2)/(2.*variance**2))


    return rates, locations

def _tbs(p):
    '''
    '''
    pulses = p['pulses']
    bursts = p['bursts']
    warmup = p['warmup']
    burst_freq = p['burst_freq']
    pulse_freq = p['pulse_freq']

    input_times = np.zeros(pulses*bursts)
    indices = np.zeros(pulses*bursts)
    cnt=-1
    for burst in range(bursts):
        for pulse in range(pulses):
            cnt+=1
            time = warmup + 1000*burst/burst_freq + 1000*pulse/pulse_freq
            input_times[cnt] = time

    print p['name']
    input_group = SpikeGeneratorGroup(1, indices, input_times*ms, name=p['name'])
    print input_group.name
    return input_group

def _poisson(p):
    '''
    '''
    print p
    if 'N' in p:
        N = p['N']
    else:
        N=1

    input_group = PoissonGroup(N, p['poisson_rates'], namespace=p)
    return input_group