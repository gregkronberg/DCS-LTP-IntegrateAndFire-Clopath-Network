'''
'''
from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy

prefs.codegen.target = 'numpy'
class Clopath2010:
    '''
    '''
    def __init__(self, ):
        '''
        '''
        self.neurons={'1':{}}
        self.synapses={'1':{}}
        self.neurons['1'] = {
            'N':1,
            'E_L':-70.6*mV,
            'g_L':30*nS,
            'delta_T':2*mV,
            'C':281*pF,
            't_noise': 20*ms,
            't_V_T':50*ms,
            'refractory':2*ms,
            'V_Trest': -50.4*mV,
            'V_Tmax':30.4*mV,
            'reset_condition':'u=-70*mV',
            'threshold_condition':'u>V_T+20*mV',
            'I_after' : 400*pA,
            'a_adapt' : 4*nS,
            'b_adapt' : 0.805*pA,
            't_w_adapt' : 144*ms,
            't_z_after' : 40*ms,
            'u_reset' : -70.6*mV,
            'I_input':0*pA,

            'u_hold':30*mV,
            'refractory_time':2*ms,
            'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
            'spike_hold_time2': 2*ms - 2*defaultclock.dt,
            't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
            'hold_spike':1,

            'rec_variables': ['u','A_LTD_homeo', 'I_nmda', 'I_gaba'],
            'rec_indices': True,

            # synapse parameters
            #====================================================================
            # ampa
            #''''''''''''''''''''

            'g_max_ampa' :50*nS,
            't_ampa' : 2*ms,
            'E_ampa' : 0*mV,
            # 'w_ampa' : 0.2,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 25*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            # 'w_nmda' : 0.5,

            # gaba
            #```````````````````````
            'g_max_gaba' : 30*nS,
            't_gaba' : 10*ms,
            'E_gaba' : -80*mV,

            # clopath
            #'''''''''''''''''''''''''
            'v_target' : 100*mV*mV,

            # visual cortex parameters from clopath 2010 table 1b
            'A_LTD' : 14E-5,#50*100E-5,
            'A_LTP' : 8E-5/ms,#50*40E-5/ms,
            'tau_lowpass2' : 7*ms,
            'tau_x' : 15*ms,
            'tau_lowpass1' : 10*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -70.6*mV,
            'theta_high' : -45.3*mV,
            'w_max_clopath' : 2,
            'x_reset':1,
            'include_homeostatic':1, # include slow homeostatic property in clopath learning rule
            }

        self.synapses['1'] = {
            'update_ampa_online':0,
            'update_nmda_online':0,  
            'updata_gaba_online':0,

            # synapse parameters
            #====================================================================
            # ampa
            #''''''''''''''''''''

            'g_max_ampa' :50*nS,
            't_ampa' : 2*ms,
            'E_ampa' : 0*mV,
            # 'w_ampa' : 0.2,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 25*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            # 'w_nmda' : 0.5,

            # gaba
            #```````````````````````
            'g_max_gaba' : 30*nS,
            't_gaba' : 10*ms,
            'E_gaba' : -80*mV,

            # vogels
            #`````````````````````````
            'tau_vogels':20*ms,
            'eta_vogels':.0001,
            'alpha_vogels':0.12,
            'w_max_vogels':2,


            # short term plasticity
            #'''''''''''''''''''''''
            'f' : 5.3,
            't_F' : 94*ms,
            'd1' : 0.45,
            't_D1' : 540*ms,
            'd2' : 0.12,
            't_D2' : 45*ms,
            'd3' : 0.98,
            't_D3' : 120E3*ms,

            # clopath
            #'''''''''''''''''''''''''
            'v_target' : 100*mV*mV,
            # visual cortex parameters from clopath 2010 table 1b
            'A_LTD' : 14E-5,#50*100E-5,
            'A_LTP' : 8E-5/ms,#50*40E-5/ms,
            'tau_lowpass2' : 7*ms,
            'tau_x' : 15*ms,
            'tau_lowpass1' : 10*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -70.6*mV,
            'theta_high' :-45.3*mV,
            'w_max_clopath' : 2,
            'x_reset':1,
            'include_homeostatic':1, # include slow homeostatic property in clopath learning rule

            # connections
            #'''''''''''''''''''''''''''
            'connect_condition':'i==1',
            'rec_variables': ['w_clopath', 'x_trace', ],
            'rec_indices': True,
        }

class Default:
    ''' default parameters
    '''
    def __init__(self, ):
        '''
        '''
        self.simulation={'1':{}}
        self.neurons={'1':{}}
        self.synapses={'1':{}}
        self.input={'1':{}}
        self.network={'1':{}}
        self.init_neurons={'1':{}}
        self.init_synapses={'1':{}}

        self.simulation = {
            'trials':1,
            'dt':0.1*ms, 
            'run_time':300*ms,
            'field_mags': [5*20*pA, 0*pA, -5*20*pA],
            'field_polarities':['anodal', 'control', 'cathodal'],
            'field_colors':['red','black','blue'],

            # variables to record
            #================================================================
            'rec_variables_neuron':['u','A_LTD_homeo', 'I_nmda', 'I_gaba'],
            'rec_variables_syn': ['w_clopath', 'x_trace', ],
        }

        self.neurons['1'] = {
            'N':1,
            'E_L':-70*mV,
            'g_L':40*nS,
            'delta_T':2*mV,
            'C':281*pF,
            't_noise': 20*ms,
            't_V_T':50*ms,
            'refractory':2*ms,
            'V_Trest': -55*mV,
            'V_Tmax':-30*mV,
            'reset_condition':'u=-70*mV',
            'threshold_condition':'u>V_T+20*mV',
            'I_after' : 400*pA,
            'a_adapt' : 4*nS,
            'b_adapt' : 0.805*pA,
            't_w_adapt' : 144*ms,
            't_z_after' : 40*ms,
            'u_reset' : -70*mV,
            'I_input':0*pA,

            'u_hold':30*mV,
            'refractory_time':2*ms,
            'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
            'spike_hold_time2': 2*ms - 2*defaultclock.dt,
            't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
            'hold_spike':1,

            'rec_variables': ['u','A_LTD_homeo', 'I_nmda', 'I_gaba'],
            'rec_indices': True,

            # synapse parameters
            #====================================================================
            # ampa
            #''''''''''''''''''''

            'g_max_ampa' :50*nS,
            't_ampa' : 2*ms,
            'E_ampa' : 0*mV,
            # 'w_ampa' : 0.2,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 25*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            # 'w_nmda' : 0.5,

            # gaba
            #```````````````````````
            'g_max_gaba' : 30*nS,
            't_gaba' : 10*ms,
            'E_gaba' : -80*mV,

            # clopath
            #'''''''''''''''''''''''''
            'v_target' : 100*mV*mV,
            'A_LTD' : 50*100E-5,
            'A_LTP' : 50*40E-5/ms,
            'tau_lowpass2' : 5*ms,
            'tau_x' : 10*ms,
            'tau_lowpass1' : 6*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -60*mV,
            'theta_high' : -50*mV,
            'w_max_clopath' : 2,
            'x_reset':1,
            'include_homeostatic':1, # include slow homeostatic property in clopath learning rule
        }

        self.synapses['1'] = {
            'update_ampa_online':0,
            'update_nmda_online':0,  
            'updata_gaba_online':0,

            # synapse parameters
            #====================================================================
            # ampa
            #''''''''''''''''''''

            'g_max_ampa' :50*nS,
            't_ampa' : 2*ms,
            'E_ampa' : 0*mV,
            # 'w_ampa' : 0.2,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 25*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            # 'w_nmda' : 0.5,

            # gaba
            #```````````````````````
            'g_max_gaba' : 30*nS,
            't_gaba' : 10*ms,
            'E_gaba' : -80*mV,

            # vogels
            #`````````````````````````
            'tau_vogels':20*ms,
            'eta_vogels':.0001,
            'alpha_vogels':0.12,
            'w_max_vogels':2,


            # short term plasticity
            #'''''''''''''''''''''''
            'f' : 5.3,
            't_F' : 94*ms,
            'd1' : 0.45,
            't_D1' : 540*ms,
            'd2' : 0.12,
            't_D2' : 45*ms,
            'd3' : 0.98,
            't_D3' : 120E3*ms,

            # clopath
            #'''''''''''''''''''''''''
            'v_target' : 100*mV*mV,
            'A_LTD' : 100E-5,
            'A_LTP' : 40E-5/ms,
            'tau_lowpass2' : 5*ms,
            'tau_x' : 10*ms,
            'tau_lowpass1' : 6*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -60*mV,
            'theta_high' : -50*mV,
            'w_max_clopath' : 2,
            'x_reset':1,
            'include_homeostatic':1, # include slow homeostatic property in clopath learning rule

            # connections
            #'''''''''''''''''''''''''''
            'connect_condition':'i==1',
            'rec_variables': ['w_clopath', 'x_trace', ],
            'rec_indices': True,
        }

        self.input['1'] = {
            # input/stimulation parameters
            #============================================================================
            'pulses' : 4,
            'bursts' : 4,
            'pulse_freq' : 100,
            'burst_freq' : 5,
            'warmup' : 10,

            'I_input':0*pA,

            'rec_variables':[],
            'rec_indices':True

        }

        self.network['1'] = {
            # network parameters
            #================================================================
            'N' : 3, 
            'syn_condition': 'i==1',
        }

        self.init_neurons['1'] = {
            'I_field':  0*pA,
            'u':self.neurons['1']['E_L'],
            'V_T':self.neurons['1']['V_Trest'],
            'w_adapt':0*pA,
            'z_after':0*pA 
        }

        self.init_synapses['1'] = {
        'F':1,
        'D1':1,
        'D2':1,
        'D3':1,
        'u_lowpass1':self.neurons['1']['E_L'],
        'u_lowpass2':self.neurons['1']['E_L'],
        'u_homeo':0*mV,
        'w_clopath':1,
        'w_vogels':0,
        'w_ampa':1, 
        'w_nmda':1,
        'w_gaba': 1,
        }


class Param:
    '''
    '''
    def __init__(self,**kwargs):
        '''
        '''
        self.p={}
        self.p_path={}
        self.init_p={}
        self.init_p_path={}

        # retrieve parameters specified by kwargs
        getattr(self, kwargs['p_type'])()
        

    def default(self, ):
        '''
        '''
        self.simulation={1:{}}
        self.neuron={'1':{}}
        self.synapse={'1':{}}
        self.input={'1':{}}
        self.network={'1':{}}
        self.init_neuron={'1':{}}
        self.init_synapse={'1':{}}

        self.simulation['1'] = {
            'trials':1,
            'dt':0.1*ms, 
            'run_time':300*ms,
            # variables to record
            #====================================================================
            'rec_variables_nrn':['u','A_LTD_homeo', 'I_nmda'],
            'rec_variables_input_syn': ['w_clopath', 'x_trace', ],
        }

        self.neurons['1'] = {
            'N':1,
            'E_L':-70*mV,
            'g_L':40*nS,
            'delta_T':2*mV,
            'C':281*pF,
            't_noise': 20*ms,
            't_V_T':50*ms,
            'refractory':2*ms,
            'V_Trest': -55*mV,
            'V_Tmax':-30*mV,
            'reset_condition':'u=-70*mV',
            'threshold_condition':'u>V_T+20*mV',
            'I_after' : 400*pA,
            'a_adapt' : 4*nS,
            'b_adapt' : 0.805*pA,
            't_w_adapt' : 144*ms,
            't_z_after' : 40*ms,
            'u_reset' : -70*mV,

            'u_hold':30*mV,
            'refractory_time':2*ms,
            'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
            'spike_hold_time2': 2*ms - 2*defaultclock.dt,
            't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
            'hold_spike':1,
        }

        self.synapses['1'] = {
            'update_ampa_online':0,
            'update_nmda_online':0,   

            # synapse parameters
            #====================================================================
            # ampa
            #''''''''''''''''''''

            'g_max_ampa' :50*nS,
            't_ampa' : 2*ms,
            'E_ampa' : 0*mV,
            'w_ampa' : 0.2,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 25*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            'w_nmda' : 0.5,

            # short term plasticity
            #'''''''''''''''''''''''
            'f' : 5.3,
            't_F' : 94*ms,
            'd1' : 0.45,
            't_D1' : 540*ms,
            'd2' : 0.12,
            't_D2' : 45*ms,
            'd3' : 0.98,
            't_D3' : 120E3*ms,

            # clopath
            #'''''''''''''''''''''''''
            'v_target' : 100*mV*mV,
            'A_LTD' : 50*100E-5,
            'A_LTP' : 50*40E-5/ms,
            'tau_lowpass2' : 5*ms,
            'tau_x' : 10*ms,
            'tau_lowpass1' : 6*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -60*mV,
            'theta_high' : -50*mV,
            'w_max_clopath' : 2,
            'x_reset':1,

            # connections
            #'''''''''''''''''''''''''''
            'connect_condition':'i==1',
        }

        self.input['1'] = {
            # input/stimulation parameters
            #============================================================================
            'pulses' : 4,
            'bursts' : 4,
            'pulse_freq' : 100,
            'burst_freq' : 5,
            'warmup' : 10,

            'I_input':0*pA,

        }

        self.network['1'] = {
            # network parameters
            #================================================================
            'N' : 3, 
            'syn_condition': 'i==1',
        }

        self.init_neurons['1'] = {
            'I_field':  0*pA,
            'u':self.p['E_L'],
            'V_T':self.p['V_Trest'],
            'w_adapt':0*pA,
            'z_after':0*pA 
        }

        self.init_synapses['1'] = {
        'F':1,
        'D1':1,
        'D2':1,
        'D3':1,
        'u_lowpass1':self.p['E_L'],
        'u_lowpass2':self.p['E_L'],
        'u_homeo':0*mV,
        'w_clopath':0.5,
        }

        self.p = {
        'trials':1,
        'dt':0.1*ms, 
        'run_time':300*ms,
        'E_L':-70*mV,
        'g_L':40*nS,
        'delta_T':2*mV,
        'C':281*pF,
        't_noise': 20*ms,
        't_V_T':50*ms,
        'refractory':2*ms,
        'V_Trest': -55*mV,
        'V_Tmax':-30*mV,
        'reset_condition':'u=-70*mV',
        'threshold_condition':'u>V_T+20*mV',
        'I_after' : 400*pA,
        'a_adapt' : 4*nS,
        'b_adapt' : 0.805*pA,
        't_w_adapt' : 144*ms,
        't_z_after' : 40*ms,
        'u_reset' : -70*mV,

        'u_hold':30*mV,
        'refractory_time':2*ms,
        'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
        'spike_hold_time2': 2*ms - 2*defaultclock.dt,
        't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
        'hold_spike':1,
        'update_ampa_online':0,
        'update_nmda_online':0,   

        # synapse parameters
        #====================================================================
        # ampa
        #''''''''''''''''''''

        'g_max_ampa' :100*nS,
        't_ampa' : 2*ms,
        'E_ampa' : 0*mV,
        'w_ampa' : 0.2,

        # nmda
        #''''''''''''''''''''''
        'g_max_nmda' : 50*nS, #g_max_ampa/2, #75*nS
        't_nmda' : 50*ms,
        'E_nmda' : 0*mV,
        'w_nmda' : 0.5,

        # short term plasticity
        #'''''''''''''''''''''''
        'f' : 5.3,
        't_F' : 94*ms,
        'd1' : 0.45,
        't_D1' : 540*ms,
        'd2' : 0.12,
        't_D2' : 45*ms,
        'd3' : 0.98,
        't_D3' : 120E3*ms,

        # clopath
        #'''''''''''''''''''''''''
        'v_target' : 100*mV*mV,
        'A_LTD' : 50*100E-5,
        'A_LTP' : 50*40E-5/ms,
        'tau_lowpass2' : 5*ms,
        'tau_x' : 10*ms,
        'tau_lowpass1' : 6*ms,
        'tau_homeo' : 1000*ms,
        'theta_low' : -60*mV,
        'theta_high' : -50*mV,
        'w_max_clopath' : 2,
        'x_reset':1,
        
        

        # input/stimulation parameters
        #============================================================================
        'pulses' : 4,
        'bursts' : 4,
        'pulse_freq' : 100,
        'burst_freq' : 5,
        'warmup' : 10,

        'I_input':0*pA,

        # network parameters
        #===================================================================
        'N' : 3, 

        'syn_condition': 'i==1',

        # variables to record
        #====================================================================
        'rec_variables_nrn':['u','A_LTD_homeo', 'I_nmda'],
        'rec_variables_input_syn': ['w_clopath', 'x_trace', ],


        }

        self.init_nrn = {
        'I_field':  0*pA,
        'u':self.p['E_L'],
        'V_T':self.p['V_Trest'],
        'w_adapt':0*pA,
        'z_after':0*pA 
        }

        self.init_input_syn={
        'F':1,
        'D1':1,
        'D2':1,
        'D3':1,
        'u_lowpass1':self.p['E_L'],
        'u_lowpass2':self.p['E_L'],
        'u_homeo':0*mV,
        'w_clopath':0.5,
        }



# functions 
#============================================================================
# design weight matrix
#''''''''''''''''''''''
def _weight_matrix_randn(Npre, Npost, w_mean, w_std,):
    ''' generate random weight matrix (pre x post) from gaussian distribution
    '''
    if Npre+Npost!=0:
        w_matrix = np.random.normal(loc=w_mean, scale=w_std, size=(Npre, Npost))
    else:
        w_matrix = []

    return w_matrix

def _weight_matrix_rand(Npre, Npost, w_min, w_max,):
    '''
    '''
    if Npre+Npost!=0:
        w_matrix = np.random.uniform(low=w_min, high=w_max, size=(Npre, Npost))
    else:
        w_matrix = []
    return w_matrix

def _weight_matrix_uniform(Npre, Npost, w):
    '''
    '''
    if Npre+Npost!=0:
        w_matrix = w*np.ones((Npre, Npost))
    else:
        w_matrix = []
    return w_matrix

def _broadcast_weight_matrix(w_matrix, i, j):
    '''
    '''
    w_vector = np.zeros(i.shape)
    for ind, val in enumerate(i):
        w_vector[ind] = w_matrix[i[ind],j[ind]]

    return w_vector


