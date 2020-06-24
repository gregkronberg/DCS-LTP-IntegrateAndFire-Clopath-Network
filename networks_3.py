from brian2 import *
import equations
import copy
import time as timer
import os
import pickle
import itertools
import uuid
import datetime
# parameters
#############################################################################
class Param(object):
    '''
    '''
    def __init__(self, **kwargs ):
        '''
        '''
        self.p={}

    def _weight_matrix_randn(self, Npre, Npost, w_mean, w_std,):
        ''' generate random weight matrix (pre x post) from gaussian distribution
        '''
        if Npre+Npost!=0:
            w_matrix = np.random.normal(loc=w_mean, scale=w_std, size=(Npre, Npost))
        else:
            w_matrix = []

        return w_matrix

    def _weight_matrix_rand(self, Npre, Npost, w_min, w_max,):
        '''
        '''
        if Npre+Npost!=0:
            w_matrix = np.random.uniform(low=w_min, high=w_max, size=(Npre, Npost))
        else:
            w_matrix = []
        return w_matrix

    def _weight_matrix_uniform(self, Npre, Npost, w):
        '''
        '''
        if Npre+Npost!=0:
            w_matrix = w*np.ones((Npre, Npost))
        else:
            w_matrix = []
        return w_matrix

    def _weight_array_uniform(self, Nsyn, w):
        '''
        '''
        w_array = w*np.ones(Nsyn)
        return w_array

    def _broadcast_weight_matrix(self, w_matrix, i, j):
        '''
        '''
        w_vector = np.zeros(i.shape)
        for ind, val in enumerate(i):
            w_vector[ind] = w_matrix[i[ind],j[ind]]

        return w_vector

class ParamLitwinKumar2014(Param):
    '''
    '''
    def __init__(self, **kwargs):
        super(ParamLitwinKumar2014, self).__init__(**kwargs)
        self.define_p()

    def define_p(self, ):
        '''
        '''
        self.simulation={1:{}}
        self.neurons={'1':{}}
        self.synapses={'1':{}}
        self.input={'1':{}}
        self.network={'1':{}}
        self.init_neurons={'1':{}}
        self.init_synapses={'1':{}}

        self.simulation['1'] = {
            'trials':1,
            'dt':0.1*ms, 
            'run_time':300*ms,
            # variables to record
            #====================================================================
            'rec_variables_nrn':['u','A_LTD_homeo', 'I_nmda'],
            'rec_variables_input_syn': ['w_clopath', 'x_trace', ],
            'spike_rec_groups':['E']
        }

        self.neurons['1'] = {
            'rec_variables':[],
            'N':1,
            'E_L':-70*mV,
            'g_L':40*nS,
            'delta_T':2*mV,
            'C':300*pF,
            't_noise': 20*ms,
            't_V_T':50*ms,
            'refractory':2*ms,
            'V_Trest': -52*mV,
            'V_Tmax':-42*mV,
            'reset_condition':'u=u_reset',
            'threshold_condition':'u>=u_thresh',
            'I_after' : 400*pA,
            'a_adapt' : 4*nS,
            'b_adapt' : 0.805*pA,
            't_w_adapt' : 144*ms,
            't_z_after' : 40*ms,
            'u_reset' : -60*mV,
            'u_max':20*mV,
            'u_thresh':-30*mV,
            'u_spike_clopath':20*mV,

            'g_max_ampa' :1*nS,
            't_ampa' : 2*ms,
            'E_ampa' : 0*mV,
            'w_ampa' : 1.8,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 1*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            'w_nmda' : 0.,

            # gaba
            #---------------
            'g_max_gaba' :1*nS,
            't_gaba' : 10*ms,
            'E_gaba' : -75*mV,
            'w_gaba' : 20.,

             # clopath
            #'''''''''''''''''''''''''
            'v_target' : 100*mV*mV,
            'A_LTD' : 0*100E-5,
            'A_LTP' : 50*40E-5/ms,
            'tau_lowpass2' : 5*ms,
            'tau_x' : 10*ms,
            'tau_lowpass1' : 6*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -60*mV,
            'theta_high' : -50*mV,
            'w_max_clopath' : 2,
            'x_reset':1,
            'include_homeostatic':0,
            'update_ampa_online':0,
            'update_nmda_online':0,  



            'u_hold':30*mV,
            'refractory_time':2*ms,
            'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
            'spike_hold_time2': 2*ms - 2*defaultclock.dt,
            't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
            'hold_spike':1,
        }
        
        self.neurons['I'] = {
            'rec_variables':[],
            'N':1,
            'E_L':-62*mV,
            'g_L':15*nS,
            'delta_T':2*mV,
            'C':300*pF,
            't_noise': 20*ms,
            't_V_T':30*ms,
            'refractory':1*ms,
            'V_Trest': -52*mV,
            'V_Tmax':-52*mV,
            'reset_condition':'u=u_reset',
            'threshold_condition':'u>=u_thresh',
            'I_after' : 0*pA,
            'a_adapt' : 0*nS,
            'b_adapt' : 0*pA,
            't_w_adapt' : 150*ms,
            't_z_after' : 40*ms,
            'u_reset' : -60*mV,
            'u_thresh' : -30*mV,
            'u_max':20*mV,
            'u_spike_clopath':20*mV,
            't_u_test':0.1*ms,

            'g_max_ampa' :1*nS,
            't_ampa' : 6*ms,
            'E_ampa' : 0*mV,
            'w_ampa' : 1.8,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 1*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            'w_nmda' : 0.,

            # gaba
            #---------------
            'g_max_gaba' :1*nS,
            't_gaba' : 2*ms,
            'E_gaba' : -75*mV,
            'w_gaba' : 20.,

             # clopath
            #'''''''''''''''''''''''''
            'v_target' : 100*mV*mV,
            'A_LTD' : 0*100E-5,
            'A_LTP' : 50*40E-5/ms,
            'tau_lowpass2' : 5*ms,
            'tau_x' : 10*ms,
            'tau_lowpass1' : 6*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -60*mV,
            'theta_high' : -50*mV,
            'w_max_clopath' : 2,
            'x_reset':1,
            'include_homeostatic':0,
            'update_ampa_online':0,
            'update_nmda_online':0,  



            'u_hold':30*mV,
            'refractory_time':2*ms,
            'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
            'spike_hold_time2': 2*ms - 2*defaultclock.dt,
            't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
            'hold_spike':1,
        }

        self.neurons['E'] = {
            'rec_variables':[],
            'N':1,
            'E_L':-70*mV,
            'g_L':15*nS,
            'delta_T':2*mV,
            'C':300*pF,
            't_noise': 20*ms,
            't_V_T':30*ms,
            'refractory':2*ms,
            'V_Trest': -52*mV,
            'V_Tmax':-42*mV,
            'reset_condition':'u=u_reset',
            'threshold_condition':'u>=u_thresh',
            # 'threshold_condition':'u==u_thresh',
            'I_after' : 0*400*pA,
            'a_adapt' : 4*nS,
            'b_adapt' : 0.805*pA,
            't_w_adapt' : 150*ms,
            't_z_after' : 40*ms,
            'u_reset' : -60*mV,
            'u_max':20*mV,
            'u_thresh':-40*mV,
            'u_spike_clopath':20*mV,
            't_u_test':0.1*ms,#self.simulation['1']['dt'],

            'g_max_ampa' :1*nS,
            't_ampa' : 6*ms,
            'E_ampa' : 0*mV,
            'w_ampa' : 1.8,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 1*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            'w_nmda' : 0.,

            # gaba
            #---------------
            'g_max_gaba' :1*nS,
            't_gaba' : 2*ms,
            'E_gaba' : -75*mV,
            'w_gaba' : 20.,

             # clopath
            #'''''''''''''''''''''''''
            'v_target' : 9*mV*mV,# 100*mV*mV,
            'A_LTD' : 8E-4,
            'A_LTP' : 14E-4/ms,
            'tau_lowpass2' : 7*ms,
            'tau_x' : 15*ms,
            'tau_lowpass1' : 10*ms,
            'tau_homeo' : 10*ms,
            'theta_low' : -70*mV,
            'theta_high' : -49*mV,
            'w_min_clopath' : 1.8,
            'w_max_clopath':21.,
            'w_init_clopath' : 2.8,
            'x_reset':1,
            'include_homeostatic':0,
            'include_normalization':1,
            'update_ampa_online':0,
            'update_nmda_online':0,  



            'u_hold':30*mV,
            'refractory_time':2*ms,
            'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
            'spike_hold_time2': 2*ms - 2*defaultclock.dt,
            't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
            'hold_spike':1,
        }

        self.synapses['1'] = {
            'rec_variables':[],
            'update_ampa_online':0,
            'update_nmda_online':0,
            'update_gaba_online':0,   

            # synapse parameters
            #====================================================================
            # ampa
            #''''''''''''''''''''

            'g_max_ampa' :1*nS,
            't_ampa' : 6*ms,
            'E_ampa' : 0*mV,
            'w_ampa' : 1.8,

            # nmda
            #''''''''''''''''''''''
            'g_max_nmda' : 1*nS, #g_max_ampa/2, #75*nS
            't_nmda' : 50*ms,
            'E_nmda' : 0*mV,
            'w_nmda' : 0.,

            # gaba
            #---------------
            'g_max_gaba' :1*nS,
            't_gaba' : 2*ms,
            'E_gaba' : -75*mV,
            'w_gaba' : 20.,

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
            'A_LTD' : 0.01*8E-4,#0.1*8E-4,
            'A_LTP' : 2.*14E-4/ms,#1.*14E-4/ms,#5*14E-4/ms,#7.5*14E-4/ms,
            'tau_lowpass2' : 7*ms,
            'tau_x' : 15*ms,
            'tau_lowpass1' : 10*ms,
            'tau_homeo' : 1000*ms,
            'theta_low' : -65*mV,
            'theta_high' : -49*mV,
            'w_min_clopath' : 1.8,
            'w_max_clopath':15.,#21.,
            'w_init_clopath' : 1.*2.8,
            'x_reset':1,
            'include_homeostatic':0,
            'include_normalization':1,
            'update_ampa_online':0,
            'update_nmda_online':0, 
            't_norm_w_clopath':1000*ms, 

            # vogels
            #---------------------
            'tau_vogels':20*ms,
            'eta_vogels':2*1E-3,#1E-5,
            'alpha_vogels':0.120,
            'w_min_vogels':49., 
            'w_max_vogels':243., 


            # connections
            #'''''''''''''''''''''''''''
            'connect_condition':'i==1',
            'delay':1*ms,
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
            'N_E' : 100, 
            'N_I': 25,
            'syn_condition': 'i==1',
        }

        self.init_neurons['1'] = {
            'I_field':  0*pA,
            'u':self.neurons['1']['u_reset'],
            'u_temp':self.neurons['1']['u_reset'],
            'V_T':self.neurons['1']['V_Trest'],
            'w_adapt':0*pA,
            'z_after':0*pA ,
            # 'A_LTD_homeo':0, 

        }
        self.init_synapses['1'] = {
            'F':1,
            'D1':1,
            'D2':1,
            'D3':1,
            'u_lowpass1':self.neurons['1']['E_L'],
            'u_lowpass2':self.neurons['1']['E_L'],
            'u_homeo':0*mV,
            'w_clopath':0.5,
            # 'saturated':0,
        }

        self.init_input_syn={
            'F':1,
            'D1':1,
            'D2':1,
            'D3':1,
            'u_lowpass1':self.neurons['1']['E_L'],
            'u_lowpass2':self.neurons['1']['E_L'],
            'u_homeo':0*mV,
            'w_clopath':0.5,
        }

class ParamZenke2015(Param):
    '''
    '''
    def __init__(self, **kwargs):
        super(ParamZenke2015, self).__init__(**kwargs)
        self.define_p()

    def define_p(self, ):
        '''
        '''
        self.simulation={'1':{}}
        self.neurons={'1':{}}
        self.synapses={'1':{}}
        self.input={'1':{}}
        self.network={'1':{}}
        self.init_neurons={'1':{}}
        self.init_synapses={'1':{}}

        self.simulation['1'] = {
            'spike_rec_groups':[],
            'trials':1,
            'dt':0.1*ms, 
            'run_time':300*ms,
            # variables to record
            #===============================================================
        }
        
        self.neurons['E_input']={
        'rec_variables':[],
        'threshold_condition':'rand()<rates*dt',
        'refractory':5*ms,
        # 'rates':10*Hz,#'E_input_timed_array(t,i)',#10*Hz, 
        'E_input_timed_array':10*Hz,
        # plasticity trace parameters
        #----------------------------
        't_istdp':20*ms,
        't_plus':20*ms,
        't_minus':20*ms,
        't_slow':100*ms,

        # short term plasticity
        #-------------------------
        't_d':200*ms,# depression time constant
        't_f':600*ms,# facilitation time constant
        'U_stp':0.2, # initial release probability parameter

        }

        self.neurons['E_rate'] = {
            'rec_variables':[],
            'N':1,
            # global rate monitor
            #---------------------
            'gamma':150,#4, # target population rate
            't_H':10*1000*ms,

            # 'G':1,
            }

        self.neurons['E'] = {
            'rec_variables':[],
            'N':1,
            # neuron equations
            'E_L':-60*mV,
            't_m':20*ms,
            'E_exc':0*mV,
            'E_gaba':-80*mV,
            'alpha':0.3, # ampa/nmda ratio
            't_ampa':5*ms,
            't_gaba':10*ms,
            't_nmda':100*ms,
            't_a':100*ms,
            'Delta_a':0.1, #adaptation strength
            't_b':20*1000*ms,
            'Delta_b':0., # 5E-4 #adaptation strength
            't_u_thresh':2*ms,
            'u_thresh_rest':-50*mV,
            'u_thresh_reset':50*mV,
            'threshold_condition':'u>=u_thresh',
            'u_reset':-60*mV,
            'refractory': 2*ms, 


            # plasticity trace parameters
            #----------------------------
            't_istdp':20*ms,
            't_plus':20*ms,
            't_minus':20*ms,
            't_slow':100*ms,

            # short term plasticity
            #-------------------------
            't_d':200*ms,# depression time constant
            't_f':600*ms,# facilitation time constant
            'U_stp':0.2, # initial release probability parameter

            # global rate monitor
            #---------------------
            'gamma':4, # target population rate

            # 'G':1,

            'consolidation_dt':1000*ms,
            'w_cons':0.,
            }

        self.neurons['I'] = {
            'rec_variables':[],
            'N':1,
            # neuron equations
            'E_L':-60*mV,
            't_m':20*ms,
            'E_exc':0*mV,
            'E_gaba':-80*mV,
            'alpha':0.3, # ampa/nmda ratio
            't_ampa':5*ms,
            't_gaba':10*ms,
            't_nmda':100*ms,
            't_a':100*ms,
            'Delta_a':0., #adaptation strength
            't_b':20*1000*ms,
            'Delta_b':0., # 5E-4 #adaptation strength
            't_u_thresh':2*ms,
            'u_thresh_rest':-50*mV,
            'u_thresh_reset':50*mV,
            'threshold_condition':'u>=u_thresh',
            'u_reset':-60*mV,
            'refractory': 2*ms, 

            # plasticity trace parameters
            #----------------------------
            't_istdp':20*ms,
            't_plus':20*ms,
            't_minus':20*ms,
            't_slow':100*ms,

            # short term plasticity
            #-------------------------
            't_d':200*ms,# depression time constant
            't_f':600*ms,# facilitation time constant
            'U_stp':0.2, # initial release probability parameter


            # global rate monitor
            #---------------------
            'gamma':4, # target population rate
            't_H':10*1000*ms,

            # 'G':1,
            }
        
        # self.neurons['global_rate'] = {
        #     't_H':10*1000*ms,
        #     'gamma':4, # target population rate
        # }

        self.synapses['E']={
            'rec_variables':[],

            # short term plasticity
            #-------------------------
            't_d':200*ms,# depression time constant
            't_f':600*ms,# facilitation time constant
            'U_stp':0.2, # initial release probability parameter

            # long term excitatory plasticity
            #---------------------------------
            'A':1.E-3, # LTP rate
            'B':1E-3, # LTD rate
            'delta':2E-5, # transmitter triggered plasticity strength
            'Beta':0.05, # heterosynaptic plasticity strength
            't_w_cons':20*60*1000*ms,
            'w_exc_plastic_min':0,
            'w_exc_plastic_max':5,
            'w_exc_plastic_init':0.1,
            'w_P':0.5,
            'P':20, # potential strength parameter for consolidation
            't_homeo':20*60*1000*ms,# metaplasiticty time constant
            't_ht':100*ms,# activity trace time constant for metaplasticity

            't_istdp':20*ms,
            't_plus':20*ms,
            't_minus':20*ms,
            't_slow':100*ms,

            'delay':1*ms, 
            # 'w_cons':0.,
            'consolidation_dt':self.neurons['E']['consolidation_dt'],

            'receptive_field_radius':8, #receptive field radius in pixels
            'n_pixels':64,

        }

        self.synapses['I']={
            'rec_variables':[],
            'w_inh_plastic_min':0,
            'w_inh_plastic_max':5,
            'w_inh_plastic_init':0.15,
            'eta':2E-5, # istdp learning rate
            'delay':1*ms, 
            'gamma':4, # target population rate

            't_istdp':20*ms,
            't_plus':20*ms,
            't_minus':20*ms,
            't_slow':100*ms,
            'w_cons':0.,
        }

        self.synapses['E_rate'] = {
            'rec_variables':[],
            't_H':10*1000*ms,
            'gamma':4, # target population rate
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
            'N_E' : 100, 
            'N_I': 25,
            'syn_condition': 'i==1',
        }

        self.init_neurons['E'] = {
            'I_field':  0*pA,
            'u':self.neurons['E']['u_reset'],
            'u_thresh':self.neurons['E']['u_thresh_rest'],
            'x_stp':1.,
            'u_stp':self.neurons['E']['U_stp']
        }
        self.init_neurons['E_rate'] = {
            # 'H':500.,
            # 'I_field':  0*pA,
            # 'u':self.neurons['E']['u_reset'],
            # 'u_thresh':self.neurons['E']['u_thresh_rest'],
        }
        self.init_neurons['E_input'] = {
            # 'rates':self.neurons['E_input']['rates']
            'x_stp':1.,
            'u_stp':self.neurons['E']['U_stp']
        }

        self.init_neurons['I'] = {
            'I_field':  0*pA,
            'u':self.neurons['I']['u_reset'],
            'u_thresh':self.neurons['I']['u_thresh_rest'],
        }

        self.init_synapses['E'] = {
            'w_exc_plastic':self.synapses['E']['w_exc_plastic_init'],
            'w_cons':0.#self.synapses['E']['w_exc_plastic_init'],
 
        }

        self.init_synapses['E_rate'] = {
            # 'w_exc_plastic':self.synapses['E']['w_exc_plastic_init'],
            # 'w_cons':self.synapses['E']['w_exc_plastic_init'],
     
        }
        self.init_synapses['I'] = {
            'w_inh_plastic':self.synapses['I']['w_inh_plastic_init']
 
        }

        # self.init_input_syn={
        #     'F':1,
        #     'D1':1,
        #     'D2':1,
        #     'D3':1,
        #     'u_lowpass1':self.neurons['1']['E_L'],
        #     'u_lowpass2':self.neurons['1']['E_L'],
        #     'u_homeo':0*mV,
        #     'w_clopath':0.5,
        # }

class Exp(object):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        self.experiment_name = self.__class__.__name__
        self.data_directory = 'Data/'+self.experiment_name+'/'
        self.figure_directory =  'png figures/'+self.experiment_name+'/'

    def _generate_trial_id(self, ):
        '''
        '''
        # create unique identifier for each trial
        uid = str(uuid.uuid1().int)[-5:]
        now = datetime.datetime.now()
        trial_id = '-'.join(['{:04d}'.format(now.year), '{:02d}'.format(now.month), '{:02d}'.format(now.day), '{:02d}'.format(now.hour), '{:02d}'.format(now.minute), '{:02d}'.format(now.second), '{:02d}'.format(now.microsecond), uid])
        return trial_id

    def _save_data(self, data, file_name=None, data_directory=None, **kwargs): # save data
        '''
        '''
        if file_name is None:
            # set file name to save data
            #----------------------------
            file_name = str(
                'data_'+
                self.experiment_name
                )
        if data_directory is None:
            data_directory=self.data_directory
        # check if folder exists with experiment name
        if os.path.isdir(data_directory) is False:
            print('making new directory to save data')
            os.mkdir(data_directory)

        # save data as pickle file
        with open(data_directory+file_name+'.pkl', 'wb') as output:
            
            print('saving data')
            pickle.dump(data, output,protocol=pickle.HIGHEST_PROTOCOL)

    def _build_spike_rec(self, brian_objects, keys, P, sim_key='1'):
        '''
        '''
        # zipped_objects = zip(keys, brian_objects)
        rec={}
        for i, key in enumerate(keys):
            rec[key] = {}
            for group_key, group in brian_objects[i].items():
                if group_key in P.simulation[sim_key]['spike_rec_groups']:
                    if 'spike_rec_N' in P.simulation[sim_key]:
                        N = P.simulation[sim_key]['spike_rec_N']
                        brian_object = group[:N]
                    else:
                        brian_object = group

                    # remove rec variables that are not in the current object
                    # rec_variables = list(set(P.__dict__[key][group_key]['rec_variables']).intersection(set(brian_object.variables.keys())))
                    
                    rec[key][group_key] = SpikeMonitor(brian_object)
        return rec

    def _build_state_rec(self, brian_objects, keys, P):
        '''
        '''
        # zipped_objects = zip(keys, brian_objects)
        rec={}
        for i, key in enumerate(keys):
            rec[key] = {}
            for group_key, group in brian_objects[i].items():
                brian_object = group

                # remove rec variables that are not in the current object
                rec_variables = list(set(P.__dict__[key][group_key]['rec_variables']).intersection(set(brian_object.variables.keys())))
                
                if 'rec_indices' not in P.__dict__[key][group_key]:
                    P.__dict__[key][group_key]['rec_indices']=True

                rec[key][group_key] = StateMonitor(brian_object, rec_variables, record=P.__dict__[key][group_key]['rec_indices'])
        return rec

        # rec = {}
        # for key in keys:
        #     rec[key]={}

        # # iterate over object types (neurons or synapses)
        # for obj_type, obj in rec.iteritems():
        #     # iterate over group
        #     for group_key, group in globals()[obj_type].iteritems():
        #         # get underlying brian object
        #         brian_object = globals()[obj_type][group_key]
        #         # setup state monitor
        #         rec[obj_type][group_key] = StateMonitor(brian_object, P.__dict__[obj_type][group_key]['rec_variables'], record=True)

    def _set_initial_conditions(self, brian_object, init_dic):
        '''
        '''
        if isinstance(brian_object, dict):
            for group_key, group in brian_object.items():
                for param, val in init_dic[group_key].items():
                    if hasattr(brian_object[group_key], param):
                        setattr(brian_object[group_key], param, val)

        else:
            for param, val in init_dic[group_key].items():
                if hasattr(brian_object, param):
                    setattr(brian_object, param, val)

    def _collect_brian_objects(self, net, *dics):
        '''
        '''
        for object_container in dics:
            for group_key, group in object_container.items():
                net.add(object_container[group_key])

        return net

    def _rec2dict(self, rec, P):
        '''
        '''
        init_dict = {
        'data':[],
        'index':[],
        'pre_index':[],
        'post_index':[],
        'brian_group_name':[],
        'group_name':[],
        'trial_id':[],
        'P':[],
        'field':[],
        }
        group_dict = {}
        # iterate over type of recorded object
        for group_type_key, group_type in rec.items():
            group_dict[group_type_key] = {}
            # iterate over groups
            for group_key, group in group_type.items():

                for var in group.record_variables:
                    if var not in group_dict[group_type_key]:
                        group_dict[group_type_key][var]=init_dict




                    group_dict[group_type_key][var]['data'].append(getattr(group, var))

                    group_dict[group_type_key][var]['index'].append(group.record)

                    if group_type_key == 'synapses':
                        pre_index = group.source.i
                        post_index = group.source.j
                    else:
                        pre_index = []
                        post_index = []

                    group_dict[group_type_key][var]['pre_index'].append(pre_index)
                    group_dict[group_type_key][var]['pre_index'].append(post_index)

                    group_dict[group_type_key][var]['brian_group_name'].append(group.source.name)

                    group_dict[group_type_key][var]['group_name'].append(group_key)

                    group_dict[group_type_key][var]['trial_id'].append(P.simulation['trial_id'])
                    group_dict[group_type_key][var]['P'].append(P)

                    group_dict[group_type_key][var]['field_mag'].append(P.simulation['field_mag'])

    def _monitor_to_dataframe(self, mon, P):
        '''
        '''
        group_data = {}

        # iterate over groups
        for group_key, group in mon.items():
            for var in group.record_variables:
                data = getattr(mon, var)
                index = group.record
                brian_group = group.source.name
                trial_id = P.simulation['trial_id']


        df = pd.DataFrame()
        for group_key, group in mon.items():
            for var in group.record_variables:
                if var not in df:
                    df[var] = [getattr(mon, var)]
                df['brian_group'] = mon.source.name

                df['index'] = mon.record

                df['trial_id']

    def _spike_monitor_to_binary_array(self, spike_monitor, run_time, t_0=0*second):
        '''
        '''
        spike_trains = spike_monitor.spike_trains()
        N = int(max(spike_trains.keys()))+1
        # print ((run_time-t_0)/defaultclock.dt)
        samples = int(np.ceil((run_time)/defaultclock.dt))
        print(samples)
        array = np.zeros((N,samples))
        for nrn_i, times in spike_trains.items():
            spike_i = np.array((times-t_0)/(defaultclock.dt), dtype=int)
            array[nrn_i, spike_i]=1

        return array

    def _build_rate_timed_array(self, N, n_episodes, n_periods, n_assemblies, n_transitions, r_rest, r_training, assembly_size, n_pre_training):
        '''
        ==Args==
        :N:
        :n_episodes:
        :n_periods:
        :n_assemblies:
        :n_transitions:
        :r_rest:
        :r_training:
        ==Return==
        '''
        # preallocate array
        rate_array = r_rest*np.ones((n_transitions,N))
        # iterate over training episodes
        for episode in range(n_episodes):
            # iterate over assemblies
            for assembly in range(n_assemblies):
                # iterate over periods (segments of simulation with time=training duration)
                for period in range(n_periods):
                    # index of the current transition
                    transition = episode*n_assemblies*n_periods + assembly*n_periods + period + n_pre_training
                    # neuron index of the current assembly ()
                    neuron_slice = slice(assembly*assembly_size, (assembly+1)*assembly_size)
                    # only update rate if the current period is a training period, otherwise keep resting rate
                    if period ==n_periods-1:
                        rate_array[transition, neuron_slice] = r_training
        return rate_array

    def _build_field_timed_array(self, field_on, field_off, field_mag):
        '''
        '''
        dt = field_on
        n_steps = field_off/field_on + 1
        field = np.zeros(int(n_steps))
        field[1:-1] = field_mag
        field_timed_array = TimedArray(field, dt=dt)
        return field_timed_array

    def _to_weight_matrix(self, synapse_group, w_key, fill_value=0):
        '''
        '''

        source_size = len(synapse_group.source)
        target_size = len(synapse_group.target)
        W = np.zeros((source_size, target_size), dtype=float)#np.full((source_size, target_size), fill_value)
        # Insert the values from the Synapses object
        if hasattr(synapse_group,w_key):
            W[synapse_group.i[:], synapse_group.j[:]] = getattr(synapse_group,w_key)[:]
        elif w_key in synapse_group.namespace:
            W[synapse_group.i[:], synapse_group.j[:]] = synapse_group.namespace[w_key]

        return W

    def _input_image(self, n_pixels=64, shape='circle', thickness=4, radius=25, r_baseline=10, r_on=5*35):
        '''
        '''
        # initialize image
        if shape=='circle':
            image = r_baseline*np.ones((n_pixels, n_pixels))
            x_vals = range(n_pixels)
            y_vals = range(n_pixels)
            x_vals, y_vals = np.mgrid[:n_pixels, :n_pixels]
            x_center = n_pixels/2
            y_center = n_pixels/2
            circle = (x_vals-x_center)**2 + (y_vals-y_center)**2
            donut = np.logical_and(circle<(radius+thickness)**2, circle>(radius-thickness)**2)
            # image[donut]=r_on
            image[donut]=r_baseline + np.random.rand(np.count_nonzero(donut))*r_on
            return image

        if shape=='square':
            image = r_baseline*np.ones((n_pixels, n_pixels))
            x_vals = range(n_pixels)
            y_vals = range(n_pixels)
            x_vals, y_vals = np.mgrid[:n_pixels, :n_pixels]
            x_center = n_pixels/2
            y_center = n_pixels/2
            square = np.abs(x_vals-x_center) + np.abs(y_vals-y_center)
            square = np.logical_and(square<=(radius+thickness), square>=(radius-thickness))
            # image[square]=r_on
            image[square]=r_baseline + np.random.rand(np.count_nonzero(square))*r_on
            return image

        if shape=='cross':
            image = r_baseline*np.ones((n_pixels, n_pixels))
            x_vals = range(n_pixels)
            y_vals = range(n_pixels)
            x_vals, y_vals = np.mgrid[:n_pixels, :n_pixels]
            x_center = n_pixels/2
            y_center = n_pixels/2
            cross =  np.abs(y_vals-y_center) - np.abs(x_vals-x_center) 
            cross = np.logical_and(cross<=(thickness), cross>=(-thickness))
            # image[cross]=r_on
            image[cross]=r_baseline + np.random.rand(np.count_nonzero(cross))*r_on
            return image

        if shape=='plus':
            image = r_baseline*np.ones((n_pixels, n_pixels))
            x_vals = range(n_pixels)
            y_vals = range(n_pixels)
            x_vals, y_vals = np.mgrid[:n_pixels, :n_pixels]
            x_center = n_pixels/2
            y_center = n_pixels/2
            # cross =  np.abs(y_vals-y_center) - np.abs(x_vals-x_center) 
            # cross = np.logical_and(cross<(thickness), cross>(-thickness))
            plus = np.logical_or(np.abs(y_vals-y_center)<thickness, np.abs(x_vals-x_center)<thickness)
            # image[plus]=r_on
            image[plus]=r_baseline + np.random.rand(np.count_nonzero(plus))*r_on
            return image

        if shape=='empty':
            image = r_baseline*np.ones((n_pixels, n_pixels))
            return image

    def _image_to_timed_array(self, images, times, image_arrays, array_name='_timedarray', dt=None, units=ms, n_pixels=64**2,):
        '''
        ==Args==
        :images:list of strings: image names
        :times: list or numpy array of transition times corresponding to the image names in images
        :image_arrays:dict of arrays: keys correspond to image names in images, values are the image arrays where each element is a firing rate
        ==Return==
        '''
        if dt is  None:
            dt = np.min(np.diff(np.array(times)))
            # FIXME check if dt has units already
            dt = dt*units

        rate_array = np.zeros((len(times), n_pixels))
        max_time = max(times)
        t_vec = np.arange(0, max_time+dt, dt)
        for i, t in enumerate(t_vec):
            time_i = np.argmin(np.abs(times-t))
            image_name = images[time_i]
            image_array = image_arrays[image_name]
            image_flattened = image_array.reshape((-1), order='F')
            rate_array[i,:]=image_flattened

        timed_rates = TimedArray(rate_array*Hz, dt=dt*units, name=array_name)
        # timed_rates = rate_array*Hz
        return timed_rates

    def _design_image_transitions(self, image_types=['empty','circle','square','cross','plus'],transitions=None, dt=10):
        '''
        '''
        image_arrays={}
        for _image_type in image_types:
            image_arrays[_image_type] = self._input_image(shape=_image_type)
        if transitions is None:
            transitions = np.zeros(n_transitions, dtype=int)
            transitions[1::2]=1
        images = [image_types[_i] for _i in transitions]
        times = [_i*dt for _i,_val in enumerate(transitions)]

        return images, times, image_arrays

    def _get_receptive_field(self, center_x, center_y, radius, n_pixels=64):

        x_vals, y_vals = np.mgrid[:2*radius, :2*radius]
        x_center = radius
        y_center = radius
        circle = (x_vals-x_center)**2 + (y_vals-y_center)**2

        x, y = np.where(circle<radius**2)
        x = x + center_x -radius
        y = y + center_y -radius

        i = x*n_pixels + y

        return x, y, i

    def _connect_synapses_from_weight_matrix(self, synapses, W, W_cons=None):
        '''
        ==Args==
        W:2D array, pre x post:
        ==Return==
        ==Comments==
        :brian assumes synapses are arranged pre x post
        '''
        # use both W and W_cons to find nonzero synapses, since some synapses may have dropped to zero but still be conencted
        # print connect_W
        if W_cons is not None:
            W_joined = np.abs(W)+np.abs(W_cons)
            connect_W_cons = np.nonzero(W_joined)
            connect = connect_W_cons
        else:
            connect = np.nonzero(W)
        # connect synapses
        synapses.connect(i=connect[0], j=connect[1])
        return synapses

    def _initialize_weights_from_matrix(self, synapses, W, W_cons=None, key='w_exc_plastic'):
        '''
        W:2D array, pre x post
        '''
        
        if W_cons is not None:
            W_joined = np.abs(W)+np.abs(W_cons)
            connect = np.nonzero(W_joined)
        else:
            connect = np.nonzero(W)
        w = getattr(synapses, key)
        w[:] = W[connect[0],connect[1]]
        return synapses

    def _double_exponential_normalization_factor(self, t_rise, t_decay):
        '''
        '''
        t_f = t_decay
        t_r = t_rise
        tp = (t_f*t_r)/(t_r - t_f) * np.log(t_r/t_f)
        factor = -np.exp(-tp/t_f) + np.exp(-tp/t_r)
        factor = 1/factor
        return factor
class exp_litwinkumar_test_2(Exp):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        super(exp_litwinkumar_test_2, self).__init__(**kwargs)

    def run(self, **kwargs):
        '''
        '''
        # directory and file name to store data
        #====================================================================
        group_data_directory = 'Datatemp/'+__name__+'/'
        group_data_filename = __name__+'_data.pkl'
        group_data_filename_train = __name__+'_data_train.pkl'
        group_data_filename_test = __name__+'_data_test.pkl'

        # load parameters
        #====================================================================
        # default. all parameter groups are initially called '1', e.g. P.neurons['1']
        P = ParamLitwinKumar2014()

        # free parameters
        # rates and weights of feedforward poisson inputs
        # timescale and set point of homeostatic plasticity
        # weights of IE synapses
        N_E = 500
        N_I = 100
        assembly_size=50
        w_EE_init=P.synapses['1']['w_init_clopath']
        w_IE_init=1.15*49.#1.*49.
        w_EI_init=1.5*1.3#1.2*1.3
        w_II_init=.9*16.2#1.*16.2
        w_FE_init=1.8
        w_FI_init=1.8
        N_assembly=3
        N_recall=1
        inputs={}
        r_input_I = 1.2*2.25E3*Hz#2.25E3*Hz
        r_input_E = 1.8*2.8E3*Hz#2.8E3*Hz#4.E3*Hz
        r_input_E_training = 3.1*r_input_E#2.3*r_input_E
        r_input_I_training =  2.4*r_input_I#2.3*r_input_I# 2.*r_input_I#4.E3*Hz + 4E3*Hz
        # training_duration = 500*ms
        # rest_duration = 2*training_duration

        # n_training_episodes = 10
        # assembly_size=50

        # temporary simulation parameters
        #--------------------------------
        reinit_W=False
        rest_train_ratio=2
        n_periods = rest_train_ratio+1
        n_episodes = 2
        n_assemblies = 5# 5
        n_post_training = 2
        n_pre_training = 1
        assembly_size = 25
        assembly_size_I = 5
        training_duration=500*ms#500*ms
        n_transitions = n_episodes*n_assemblies*n_periods+n_post_training+n_pre_training
        sim_duration = n_transitions*training_duration

        for _assembly in range(n_assemblies):
            P.network[str(_assembly)]={
            'assembly_index':slice(_assembly*assembly_size, (_assembly+1)*assembly_size)
            } 
        # r_input_I = 2.25E3*Hz
        # r_input_E = 2.5E3*Hz


        # load equations for adaptive exponential integrate and fire neuron
        #------------------------------------------------------------------
        Eq = equations.AdexLitwinKumar2014()

        neurons={}
        synapses={}

        # excitatory neurons
        #-------------------------------------------------------------------
        neuron_group='E'
        P.neurons[neuron_group]=copy.deepcopy(P.neurons['E'])
        P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
        P.neurons[neuron_group]['N']=N_E
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
            )

        # inhibitory neurons
        #-----------------------------------------------------------
        neuron_group='I'
        P.neurons[neuron_group]=copy.deepcopy(P.neurons['I'])
        P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
        P.neurons[neuron_group]['N']=N_I
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
            )

        # parameter updates to be distributed to all synapses
        #----------------------------------------------------------
        P.synapses['1']['update_ampa_online']=0
        P.synapses['1']['update_gaba_online']=0
        P.synapses['1']['update_nmda_online']=0

        # initialize trained weight matrices from previous runs
        #-----------------------------------------------------
        data_directory = '_Data/'+self.experiment_name+'/'
        file_name_trained = 'W_trained.pkl'
        file_name_assemblies = 'W_assemblies.pkl'
        if reinit_W:
            with open(data_directory+file_name_trained, 'rb') as pkl_file:
                W_pretrained = pickle.load(pkl_file)
            with open(data_directory+file_name_assemblies, 'rb') as pkl_file:
                W_pretrained_assemblies = pickle.load(pkl_file)

        # recurrent EE synapses
        ###################################################################
        synapse_group='EE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.synapses[synapse_group]['update_ampa_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'i!=j'
        P.synapses[synapse_group]['connect_p'] =  0.2
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['E'], Eq.synapse_e_recurrent, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method='euler', delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        if reinit_W:
            synapses[synapse_group].connect(i=W_pretrained['EE']['i'], j=W_pretrained['EE']['j'])
        else:
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
        # print('i',W_pretrained['EE']['i'][:20])
        # print('j',W_pretrained['EE']['j'][:20])
        

        # initialize uniform weights 
        #-----------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_ampa'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)
        P.init_synapses[synapse_group]['w_clopath'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)


        # recurrent EI synapses
        #####################################################################
        synapse_group='EI'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.synapses[synapse_group]['update_ampa_online']=0
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  0.2
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['I'], Eq.synapse_e_recurrent, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method='euler', delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
        # initialize uniform weights 
        #-----------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_ampa'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EI_init)
        P.init_synapses[synapse_group]['w_clopath'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EI_init)

        # recurrent IE synapses
        #####################################################################
        synapse_group='IE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.synapses[synapse_group]['update_gaba_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  0.2
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['I'], neurons['E'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method='euler', delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        if reinit_W:
            synapses[synapse_group].connect(i=W_pretrained[synapse_group]['i'], j=W_pretrained[synapse_group]['j'])
        else:
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])

        # initialize uniform weights 
        #---------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_gaba'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_IE_init)
        P.init_synapses[synapse_group]['w_vogels'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_IE_init)

        # recurrent II synapses
        #####################################################################
        synapse_group='II'
        # copy default parameters
        #---------------------------
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.synapses[synapse_group]['update_gaba_online']=0
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connection conditions
        #-----------------------
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  0.2 
        # make synapses
        #------------------
        sparams = P.synapses[synapse_group]
        synapses[synapse_group] = Synapses(neurons['I'], neurons['I'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method='euler', delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        #------------------
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
        # initialize uniform weights 
        #---------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_gaba'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_II_init)
        P.init_synapses[synapse_group]['w_vogels'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_II_init)

        # feedforward synapses on to E neurons during training
        ####################################################################
        # inputs['E'] = PoissonGroup(N_E,r_input_E)
        rate_array = self._build_rate_timed_array(N=N_E, n_episodes=n_episodes, n_periods=n_periods, n_assemblies=n_assemblies, n_transitions=n_transitions, r_rest=r_input_E, r_training=r_input_E_training, assembly_size=assembly_size, n_pre_training=n_pre_training)

        timed_rates = TimedArray(rate_array, dt=training_duration)
        inputs['E'] = PoissonGroup(N_E,rates='timed_rates(t,i)')
        # recurrent EE synapses
        synapse_group='FE_train'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] = 'i==j'
        # synapse parameters
        
        # create synapses
        synapses[synapse_group] = Synapses(inputs['E'], neurons['E'], Eq.synapse_e_feedforward, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method='euler')
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])
        # initialize uniform weights 
        #```````````````````````````
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_ampa'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FE_init)
        P.init_synapses[synapse_group]['w_clopath'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FE_init)

        # feedforward synapses on to I neurons during training
        ####################################################################
        rate_array_I = self._build_rate_timed_array(N=N_I, n_episodes=n_episodes, n_periods=n_periods, n_assemblies=n_assemblies, n_transitions=n_transitions, r_rest=r_input_I, r_training=r_input_I_training, assembly_size=assembly_size_I, n_pre_training=n_pre_training)

        timed_rates_I = TimedArray(rate_array_I, dt=training_duration)
        inputs['I'] = PoissonGroup(N_I,rates='timed_rates_I(t,i)')
        # inputs['I']= PoissonGroup(N_I, r_input_I)
        # recurrent EE synapses
        synapse_group='FI_train'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] = 'i==j'
        # synapse parameters
        
        # create synapses
        synapses[synapse_group] = Synapses(inputs['I'], neurons['I'], Eq.synapse_e_feedforward, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method='euler')
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])
        

        # initialize uniform weights 
        #```````````````````````````
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_ampa'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FI_init)
        P.init_synapses[synapse_group]['w_clopath'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FI_init)


        # initial conditions
        #####################################################################
        
        # P.init_synapses[synapse_group]['w_clopath'] = W_pretrined['EE']

        # P.init_synapses['2']=copy.deepcopy(P.init_synapses['1'])
        self._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
        self._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

        if reinit_W:
            synapses['EE'].w_clopath[:] = W_pretrained['EE']['W'][(synapses['EE'].i[:],synapses['EE'].j[:])]
            print(W_pretrained['EE']['W'][:50,:50])
            synapses['IE'].w_vogels[:] = W_pretrained['IE']['W'][synapses['IE'].i[:],synapses['IE'].j[:]]
            print(W_pretrained['IE']['W'][:10,:10])
        # set up recording
        #####################################################################
        # recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
        # state monitors
        #-----------------
        # P.neurons['E']['rec_variables'].extend(['I_exp','u','u_test','A_LTD_homeo','w_clopath_total_recurrent'])
        # P.neurons['E']['rec_indices'] = range(10)
        # P.neurons['I']['rec_variables'].extend(['u_test'])
        # P.neurons['I']['rec_indices'] = range(10)
        # P.synapses['EE']['rec_variables'].extend(['w_clopath',])
        # P.synapses['EE']['rec_indices'] = synapses['EE'][slice(0,100),slice(0,100)]
        # P.synapses['IE']['rec_variables'].extend(['w_vogels',])
        # P.synapses['IE']['rec_indices'] = synapses['IE'][slice(0,100),slice(0,100)]
        self.rec = self._build_state_rec(brian_objects=[neurons, synapses,], keys=['neurons', 'synapses',], P=P)
        # spike monitors
        #--------------------- 
        P.simulation['1']['spike_rec_groups']=['E']
        # spike recording
        self.spike_rec = self._build_spike_rec(brian_objects=[neurons], keys=['neurons'], P=P)
        # self.spike_rec = SpikeMonitor(inputs['I'])
        # population rate monitors
        #--------------------------
        self.rate_rec={}
        self.rate_rec['assembly']=PopulationRateMonitor(neurons['E'][slice(assembly_size)])
        # self.rate_rec['other']=PopulationRateMonitor(neurons['E'][assembly_size:2*assembly_size])

        # set up network
        #####################################################################
        self.net = Network()
        # make sure you collect 
        self.net = self._collect_brian_objects(self.net, inputs, neurons, synapses, self.rec['neurons'], self.rec['synapses'], self.spike_rec['neurons'], self.rate_rec)
        
        # self.net.add(object_container[group_key])
        # self.net.remove(*objs)
        # set up simulation parameters
        #####################################################################
        # set time step
        prefs.codegen.target='numpy'
        defaultclock.dt = P.simulation['1']['dt']
        P.simulation['1']['run_time'] = sim_duration
        
        # apply field
        ####################################################################
        # field_timed_array = self._build_field_timed_array(field_on=10*ms, field_off=100*ms, field_mag=100*pA)

        # P.neurons['E']['I_field'] = 'field_timed_array(t)'
        # neurons['E'].I_field = P.neurons['E']['I_field']

        # run simulation
        ###################################################################
        # regular run
        #-------------
        # start_timer = timer.time()
        # self.net.run(P.simulation['1']['run_time'])
        
        # end_timer = timer.time()
        # print('run duration:', str(end_timer-start_timer))

        # FIXME set as attributes
        #-------------------------
        self.P = P
        self.inputs=inputs
        self.synapses=synapses
        self.neurons=neurons
        self.timed_rates = timed_rates

        # dump weights
        #------------------------------------------------------------
        # data directories and file names
        #-------------------------------------
        data_directory = '_Data/'+self.experiment_name+'/'
        n_dumps = n_episodes
        run_time_dump = P.simulation['1']['run_time']/n_dumps
        # preallocate array for mean 
        W_out_pairs = itertools.product(P.network.keys(), repeat=2)
        if reinit_W:
            for _pair in W_pretrained_assemblies:
                W_pretrained_assemblies[_pair]['mean'] = np.append(W_pretrained_assemblies[_pair]['mean'], np.zeros(n_dumps))
                W_pretrained_assemblies[_pair]['std'] = np.append(W_pretrained_assemblies[_pair]['std'], np.zeros(n_dumps))

            self.W_assemblies = W_pretrained_assemblies
        else:
            self.W_assemblies={}
            for _pair in W_out_pairs:
                _assemblies = '_'.join(_pair)
                self.W_assemblies[_assemblies] = {
                        'mean':np.zeros(n_dumps),
                        'std':np.zeros(n_dumps),}
        #     sl
        self.W_trained={'EE':{}, 'IE':{}}
        if reinit_W:
            self.W_trained['EE']['W_old'] = W_pretrained['EE']['W']
            self.W_trained['IE']['W_old'] = W_pretrained['IE']['W']

        start_timer = timer.time()
        for dump in range(n_dumps):
            # run
            #--------------------------
            self.net.run(run_time_dump)

            # get weight matrix 
            #----------------------
            self.W_trained['EE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_clopath',)
            self.W_trained['IE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['IE'], w_key='w_vogels',)

            self.W_trained['EE']['i'] = list(self.synapses['EE'].i)
            self.W_trained['EE']['j'] = list(self.synapses['EE'].j)
            self.W_trained['IE']['i'] = list(self.synapses['IE'].i)
            self.W_trained['IE']['j'] = list(self.synapses['IE'].j)

            # iterate over assembly combinations
            #-------------------------------------
            print('assemblies',self.W_assemblies.keys())
            for _assemblies in self.W_assemblies.keys():

                _pre = _assemblies.split('_')[0]
                _post = _assemblies.split('_')[1]
                _pre_slice = P.network[_pre]['assembly_index']
                _post_slice = P.network[_post]['assembly_index']
                _W = self.W_trained['EE']['W'][_pre_slice, _post_slice]
                self.W_assemblies[_assemblies]['mean'][dump-n_dumps] = np.mean(_W[np.nonzero(_W)])
                self.W_assemblies[_assemblies]['std'][dump-n_dumps] = np.std(_W[np.nonzero(_W)])

            # iterate over assemblies and store mean weights
            #------------------------------------------------
            # for _assembly, _vals in P.network.items():
            #     assembly_slice = _vals['assembly_index']
            #     # self.W_mean_assembly[_assembly][dump] = np.mean(self.synapses['EE'][assembly_slice].w_clopath)
            #     _W = self.W_trained['EE'][assembly_slice, assembly_slice]
            #     self.W_in[_assembly][dump] = np.mean(_W[np.nonzero(_W)])
        print(self.W_trained['EE']['W'][:50,:50])
        file_name = 'W_assemblies'
        self._save_data(data=self.W_assemblies, file_name=file_name, data_directory=data_directory)
        file_name = 'W_trained'
        self._save_data(data=self.W_trained, file_name=file_name, data_directory=data_directory)

        end_timer = timer.time()
        print('run duration:', str(end_timer-start_timer))
        # # store initialized network state
        # net.store('initial')

        # # dictionary for group data over multiple trials
        # train_group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename_train, df=True)
        # test_group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename_test, df=True)

        # # FIXME 
        # # FOR REPEATED SIMULATIONS ARE POISSON INPUTS REGENEREATED?
        # # FREEZE WEIGHTS AND REACTIVATE NEURONS AFTER TRAIN

        # P.simulation['run_time'] = steps*step_dt
        # # set number of trials
        # P.simulation['trials']=1
        # for trial in range(P.simulation['trials']):
        #     # restore initial conditions after each trial
        #     net.restore('initial')

        #     # Training
        #     #====================================================================
        #     # set ampa weights to be plastic
        #     # synapses['EE'].update_ampa_online =1
        #     P.synapses['EE']['update_ampa_online']=1
        #     P.synapses['FE_train']['update_ampa_online']=1
        #     P.synapses['FE_test']['update_ampa_online']=0
            
        #     # create two timed arrays, one for training and one for test
        #     # when training, the test array is all zeros and vice versa
        #     # reshuffle timed arrray values
        #     active_arrays={}
        #     inactive_arrays={}
        #     np.random.shuffle(t_array)
        #     rate_array = rate*t_array.T
        #     field_pair_i = 5
        #     field_array = P.simulation['field_mags'][P.simulation['field_polarities'].index('anodal')]*t_array[field_pair_i,:].T
        #     active_arrays['input_timed_array'] = TimedArray(rate_array*Hz, dt=step_dt)
        #     active_arrays['field_timed_array'] = TimedArray(field_array, dt=step_dt)
        #     inactive_arrays['input_timed_array'] = TimedArray(np.zeros(rate_array.shape)*Hz, dt=step_dt)
        #     inactive_arrays['field_timed_array'] = TimedArray(np.zeros(field_array.shape)*Hz, dt=step_dt)

        #     # print active_arrays['field_timed_array'].values
        #     # make sure timed arrays are available to the appropriate namespace
        #     P.input['FF_train']['input_timed_array'] = active_arrays['input_timed_array']
        #     P.input['FF_test']['input_timed_array'] = inactive_arrays['input_timed_array']
        #     P.neurons['E']['field_timed_array'] = active_arrays['field_timed_array']
            
        #     # store randomized initial condition
        #     net.store('randomized')
            
        #     # generate unique trial id
        #     P.simulation['trial_id'] = str(uuid.uuid4())

        #     # set electric field in parameter dictionaries
        #     P.simulation['field_mag'] = P.simulation['field_mags'][P.simulation['field_polarities'].index('anodal')]

        #     P.simulation['field_polarity'] = 'anodal'
        #     P.simulation['field_color'] = P.simulation['field_colors'][P.simulation['field_polarities'].index('anodal')]

        #     # FIX
        #     # FIXME check if neuron objects have access to top namespace 
        #     # only add field to excitatory neurons 
        #     P.neurons['E']['I_field'] = 'field_timed_array(t)'
        #     neurons['E'].I_field = P.neurons['E']['I_field']

        #     net.run(P.simulation['run_time'])

        #     print 'first run finished'

        #     # get trained weights
        #     trained_weights = {}
        #     weight_keys = ['w_ampa','w_nmda', 'w_gaba', 'w_clopath','w_vogels']
        #     for syn_group, syn in synapses.iteritems():
        #         trained_weights[syn_group]={}
        #         for weight_key in weight_keys:
        #             if hasattr(syn, weight_key):
        #                 trained_weights[syn_group][weight_key] = getattr(syn, weight_key)[-1]

        #     # training data
        #     train_df = analysis._rec2df(rec=rec, P=P, include_P=False)

        #     # Test
        #     #==================================================================
        #     # restore randomized network
        #     net.restore('randomized')

        #     # set ampa weights to be fixed
        #     # synapses['EE'].update_ampa_online = 0
        #     P.synapses['EE']['update_ampa_online']=0
        #     P.synapses['FE_train']['update_ampa_online']=0
        #     P.synapses['FE_test']['update_ampa_online']=0

        #     print synapses['EE'].namespace
        #     # initialize weights to trained values
        #     for syn_group, syn in synapses.iteritems():
        #         if hasattr(syn, 'w_ampa'):
        #             synapses[syn_group].w_ampa=trained_weights[syn_group]['w_clopath']
        #             synapses[syn_group].w_clopath=trained_weights[syn_group]['w_clopath']
        #         if hasattr(syn, 'w_gaba'):
        #             synapses[syn_group].w_gaba=trained_weights[syn_group]['w_vogels']
        #             synapses[syn_group].w_vogels=trained_weights[syn_group]['w_vogels']

        #     # FIXME ACTIVATE SEPARATE POISSON INPUTS FOR TEST PHASE
        #     # make sure timed arrays are available to the appropriate namespace
        #     P.input['FF_train']['input_timed_array'] = inactive_arrays['input_timed_array']
        #     P.input['FF_test']['input_timed_array'] = active_arrays['input_timed_array']
        #     P.neurons['E']['field_timed_array'] = inactive_arrays['field_timed_array']

        #     # run simulation
        #     net.run(P.simulation['run_time'])

        #     # convert recorded data to pandas dataframe
        #     test_df = analysis._rec2df(rec=rec, P=P, include_P=False)

        #     # add to group data
        #     train_group_df = train_group_df.append(train_df, ignore_index=True)
        #     test_group_df = test_group_df.append(test_df, ignore_index=True)


        # # save data
        # train_group_df.to_pickle(group_data_directory+group_data_filename_train)
        # test_group_df.to_pickle(group_data_directory+group_data_filename_test)

class exp_zenke_test(Exp):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        super(exp_zenke_test, self).__init__(**kwargs)

    def run(self, **kwargs):
        '''
        '''
        # directory and file name to store data
        #====================================================================
        group_data_directory = 'Datatemp/'+__name__+'/'
        group_data_filename = __name__+'_data.pkl'
        group_data_filename_train = __name__+'_data_train.pkl'
        group_data_filename_test = __name__+'_data_test.pkl'

        # load parameters
        #====================================================================
        # default. all parameter groups are initially called '1', e.g. P.neurons['1']
        P = ParamZenke2015()

        # free parameters
        # rates and weights of feedforward poisson inputs
        # timescale and set point of homeostatic plasticity
        # weights of IE synapses
        N_E = 4000
        N_I = 1000
        p_connect = 0.1
        assembly_size=50
        w_EE_init=0.1
        w_IE_init=1*0.15
        w_EI_init=0.2
        w_II_init=0.2
        w_FE_init=0.5
        w_FI_init=2.5
        N_assembly=3
        N_recall=1
        inputs={}
        r_input_I = .1*2.25E3*Hz#2.25E3*Hz
        r_input_E = 0.2*1.8*2.8E3*Hz#2.8E3*Hz#4.E3*Hz
        r_input_E_training = 3.1*r_input_E#2.3*r_input_E
        r_input_I_training =  2.4*r_input_I#2.3*r_input_I# 2.*r_input_I#4.E3*Hz + 4E3*Hz
        # training_duration = 500*ms
        # rest_duration = 2*training_duration

        # n_training_episodes = 10
        # assembly_size=50

        # temporary simulation parameters
        #--------------------------------
        reinit_W=False
        rest_train_ratio=2
        n_periods = rest_train_ratio+1
        n_episodes = 2
        n_assemblies = 5# 5
        n_post_training = 2
        n_pre_training = 1
        assembly_size = 5
        assembly_size_I = 2
        training_duration=500*ms#500*ms
        n_transitions = n_episodes*n_assemblies*n_periods+n_post_training+n_pre_training
        sim_duration = n_transitions*training_duration

        for _assembly in range(n_assemblies):
            P.network[str(_assembly)]={
            'assembly_index':slice(_assembly*assembly_size, (_assembly+1)*assembly_size)
            } 
        # r_input_I = 2.25E3*Hz
        # r_input_E = 2.5E3*Hz


        # load equations for adaptive exponential integrate and fire neuron
        #------------------------------------------------------------------
        Eq = equations.Zenke2015()

        neurons={}
        synapses={}

        # excitatory neurons
        #-------------------------------------------------------------------
        neuron_group='E'
        P.neurons[neuron_group]['N']=N_E
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_e, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams, 
            events={'consolidate':Eq.neuron_consolidation_event}
            )

        
        # inhibitory neurons
        #-----------------------------------------------------------
        neuron_group='I'
        P.neurons[neuron_group]['N']=N_I
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_i, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
            )

        # connect to I neurons to alter plasticity
        #----------------------------------------
        # synapse_group='GRI' # global rate to inhibitory
        # # copy default parameters
        # P.synapses[synapse_group]=copy.deepcopy(P.synapses['global_rate'])
        # P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # # connect all to all
        # P.synapses[synapse_group]['connect_condition'] =  'True'
        # # synapse parameters
        # sparams = P.synapses[synapse_group]
        # # make synapses
        # synapses[synapse_group] = Synapses(neurons['E'], neurons['I'], Eq.synapse_global_rate, on_pre=Eq.synapse_global_rate_pre, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, )
        # # connect synapses
        # #--------------------
        # synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])

        # parameter updates to be distributed to all synapses
        #----------------------------------------------------------
        # P.synapses['1']['update_ampa_online']=0
        # P.synapses['1']['update_gaba_online']=0
        # P.synapses['1']['update_nmda_online']=0

        # initialize trained weight matrices from previous runs
        #-----------------------------------------------------
        data_directory = '_Data/'+self.experiment_name+'/'
        file_name_trained = 'W_trained.pkl'
        file_name_assemblies = 'W_assemblies.pkl'
        if reinit_W:
            with open(data_directory+file_name_trained, 'rb') as pkl_file:
                W_pretrained = pickle.load(pkl_file)
            with open(data_directory+file_name_assemblies, 'rb') as pkl_file:
                W_pretrained_assemblies = pickle.load(pkl_file)

        # recurrent EE synapses
        ###################################################################
        synapse_group='EE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.synapses[synapse_group]['update_ampa_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'i!=j'
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        # synapses[synapse_group] = Synapses(neurons['E'], neurons['E'], Eq.synapse_e, on_pre=Eq.synapse_e_pre, on_post=Eq.synapse_e_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        synapses[synapse_group] = Synapses(neurons['E'], neurons['E'], Eq.synapse_e, 
            on_pre={'pre':Eq.synapse_e_pre,'consolidation':Eq.synapse_consolidate_on_event}, 
            on_event={'pre':'spike', 'consolidation':'consolidate'},
            on_post=Eq.synapse_e_post, 
            namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        #--------------------
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
        # if reinit_W:
        #     synapses[synapse_group].connect(i=W_pretrained['EE']['i'], j=W_pretrained['EE']['j'])
        # else:
        #     synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])

        # initialize uniform weights 
        #-----------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)

        # recurrent EI synapses
        #####################################################################
        synapse_group='EI'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.synapses[synapse_group]['update_ampa_online']=0
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['I'], Eq.synapse_e, on_pre=Eq.synapse_e_pre, on_post=Eq.synapse_e_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
        # initialize uniform weights 
        #-----------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EI_init)

        # recurrent IE synapses
        #####################################################################
        synapse_group='IE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['I'])
        P.synapses[synapse_group]['update_gaba_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['I'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['I'], neurons['E'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        if reinit_W:
            synapses[synapse_group].connect(i=W_pretrained[synapse_group]['i'], j=W_pretrained[synapse_group]['j'])
        else:
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])

        # initialize uniform weights 
        #---------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_inh_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_IE_init)

        # recurrent II synapses
        #####################################################################
        synapse_group='II'
        # copy default parameters
        #---------------------------
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['I'])
        P.synapses[synapse_group]['update_gaba_online']=0
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['I'])
        # connection conditions
        #-----------------------
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        P.synapses[synapse_group]['eta'] =  0. 
        # make synapses
        #------------------
        sparams = P.synapses[synapse_group]
        synapses[synapse_group] = Synapses(neurons['I'], neurons['I'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # connect synapses
        #------------------
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
        # initialize uniform weights 
        #---------------------------
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_inh_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_II_init)


        # feedforward synapses on to E neurons during training
        ####################################################################
        # inputs['E'] = PoissonGroup(N_E,r_input_E)
        rate_array = self._build_rate_timed_array(N=N_E, n_episodes=n_episodes, n_periods=n_periods, n_assemblies=n_assemblies, n_transitions=n_transitions, r_rest=r_input_E, r_training=r_input_E_training, assembly_size=assembly_size, n_pre_training=n_pre_training)

        timed_rates = TimedArray(rate_array, dt=training_duration)
        inputs['E'] = PoissonGroup(N_E,rates='timed_rates(t,i)')
        # recurrent EE synapses
        synapse_group='FE_train'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] = 'i==j'
        # synapse parameters
        
        # create synapses
        synapses[synapse_group] = Synapses(inputs['E'], neurons['E'], Eq.synapse_e_static, on_pre=Eq.synapse_e_pre_static, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])
        # initialize uniform weights 
        #```````````````````````````
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FE_init)

        # feedforward synapses on to I neurons during training
        ####################################################################
        rate_array_I = self._build_rate_timed_array(N=N_I, n_episodes=n_episodes, n_periods=n_periods, n_assemblies=n_assemblies, n_transitions=n_transitions, r_rest=r_input_I, r_training=r_input_I_training, assembly_size=assembly_size_I, n_pre_training=n_pre_training)

        timed_rates_I = TimedArray(rate_array_I, dt=training_duration)
        inputs['I'] = PoissonGroup(N_I,rates='timed_rates_I(t,i)')
        # inputs['I']= PoissonGroup(N_I, r_input_I)
        # recurrent EE synapses
        synapse_group='FI_train'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] = 'i==j'
        # synapse parameters
        
        # create synapses
        synapses[synapse_group] = Synapses(inputs['I'], neurons['I'], Eq.synapse_e_static, on_pre=Eq.synapse_e_pre_static, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])

        # initialize uniform weights 
        #```````````````````````````
        # number of synapses
        Nsyn = len(synapses[synapse_group].i)
        # initial weight matrices
        P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FI_init)


        # initial conditions
        #####################################################################
        
        # P.init_synapses[synapse_group]['w_clopath'] = W_pretrined['EE']

        # P.init_synapses['2']=copy.deepcopy(P.init_synapses['1'])
        self._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
        self._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

        # if reinit_W:
        #     synapses['EE'].w_clopath[:] = W_pretrained['EE']['W'][(synapses['EE'].i[:],synapses['EE'].j[:])]
        #     print(W_pretrained['EE']['W'][:50,:50])
        #     synapses['IE'].w_vogels[:] = W_pretrained['IE']['W'][synapses['IE'].i[:],synapses['IE'].j[:]]
        #     print(W_pretrained['IE']['W'][:10,:10])
        # set up recording
        #####################################################################
        # recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
        # state monitors
        #-----------------
        # P.neurons['E']['rec_variables'].extend(['u'])
        # P.neurons['E']['rec_indices'] = range(10)
        # P.neurons['I']['rec_variables'].extend(['u'])
        # P.neurons['I']['rec_indices'] = range(10)
        P.synapses['EE']['rec_variables'].extend(['w_exc_plastic','w_cons'])
        P.synapses['EE']['rec_indices'] = synapses['EE'][slice(0,10),slice(0,10)]
        # P.synapses['IE']['rec_variables'].extend(['w_inh_plastic',])
        # P.synapses['IE']['rec_indices'] = synapses['IE'][slice(0,N_I),slice(0,N_I)]
        self.rec = self._build_state_rec(brian_objects=[neurons, synapses,], keys=['neurons', 'synapses',], P=P)
        # spike monitors
        #--------------------- 
        # P.simulation['1']['spike_rec_groups']=['E']
        P.simulation['1']['spike_rec_groups']=[]
        # spike recording
        self.spike_rec = self._build_spike_rec(brian_objects=[neurons], keys=['neurons'], P=P)
        # self.spike_rec = SpikeMonitor(inputs['I'])
        # population rate monitors
        #--------------------------
        self.rate_rec={}
        # self.rate_rec['assembly']=PopulationRateMonitor(neurons['E'][slice(assembly_size)])
        # self.rate_rec['other']=PopulationRateMonitor(neurons['E'][assembly_size:2*assembly_size])

        # set up network
        #####################################################################
        self.net = Network()
        # make sure you collect 
        self.net = self._collect_brian_objects(self.net, inputs, neurons, synapses, self.rec['neurons'], self.rec['synapses'], self.spike_rec['neurons'], self.rate_rec)
        
        # self.net.add(object_container[group_key])
        # self.net.remove(*objs)
        # set up simulation parameters
        #####################################################################
        # set time step
        # prefs.codegen.target='numpy'
        # prefs.codegen.target='cython'

        defaultclock.dt = P.simulation['1']['dt']
        P.simulation['1']['run_time'] = sim_duration
        
        # apply field
        ####################################################################
        # field_timed_array = self._build_field_timed_array(field_on=10*ms, field_off=100*ms, field_mag=100*pA)

        # P.neurons['E']['I_field'] = 'field_timed_array(t)'
        # neurons['E'].I_field = P.neurons['E']['I_field']

        # run simulation
        ###################################################################
        # regular run
        #-------------
        start_timer = timer.time()
        self.net.run(P.simulation['1']['run_time'])
        
        end_timer = timer.time()
        print('run duration:', str(end_timer-start_timer))

        # FIXME set as attributes
        #-------------------------
        self.P = P
        self.inputs=inputs
        self.synapses=synapses
        self.neurons=neurons
        self.timed_rates = timed_rates

        # dump weights
        #------------------------------------------------------------
        # data directories and file names
        #-------------------------------------
        # data_directory = '_Data/'+self.experiment_name+'/'
        # n_dumps = n_episodes
        # run_time_dump = P.simulation['1']['run_time']/n_dumps
        # # preallocate array for mean 
        # W_out_pairs = itertools.product(P.network.keys(), repeat=2)
        # if reinit_W:
        #     for _pair in W_pretrained_assemblies:
        #         W_pretrained_assemblies[_pair]['mean'] = np.append(W_pretrained_assemblies[_pair]['mean'], np.zeros(n_dumps))
        #         W_pretrained_assemblies[_pair]['std'] = np.append(W_pretrained_assemblies[_pair]['std'], np.zeros(n_dumps))

        #     self.W_assemblies = W_pretrained_assemblies
        # else:
        #     self.W_assemblies={}
        #     for _pair in W_out_pairs:
        #         _assemblies = '_'.join(_pair)
        #         self.W_assemblies[_assemblies] = {
        #                 'mean':np.zeros(n_dumps),
        #                 'std':np.zeros(n_dumps),}
        # #     sl
        # self.W_trained={'EE':{}, 'IE':{}}
        # if reinit_W:
        #     self.W_trained['EE']['W_old'] = W_pretrained['EE']['W']
        #     self.W_trained['IE']['W_old'] = W_pretrained['IE']['W']

        # start_timer = timer.time()
        # for dump in range(n_dumps):
        #     # run
        #     #--------------------------
        #     self.net.run(run_time_dump)

        #     # get weight matrix 
        #     #----------------------
        #     self.W_trained['EE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_clopath',)
        #     self.W_trained['IE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['IE'], w_key='w_vogels',)

        #     self.W_trained['EE']['i'] = list(self.synapses['EE'].i)
        #     self.W_trained['EE']['j'] = list(self.synapses['EE'].j)
        #     self.W_trained['IE']['i'] = list(self.synapses['IE'].i)
        #     self.W_trained['IE']['j'] = list(self.synapses['IE'].j)

        #     # iterate over assembly combinations
        #     #-------------------------------------
        #     print('assemblies',self.W_assemblies.keys())
        #     for _assemblies in self.W_assemblies.keys():

        #         _pre = _assemblies.split('_')[0]
        #         _post = _assemblies.split('_')[1]
        #         _pre_slice = P.network[_pre]['assembly_index']
        #         _post_slice = P.network[_post]['assembly_index']
        #         _W = self.W_trained['EE']['W'][_pre_slice, _post_slice]
        #         self.W_assemblies[_assemblies]['mean'][dump-n_dumps] = np.mean(_W[np.nonzero(_W)])
        #         self.W_assemblies[_assemblies]['std'][dump-n_dumps] = np.std(_W[np.nonzero(_W)])

        #     # iterate over assemblies and store mean weights
        #     #------------------------------------------------
        #     # for _assembly, _vals in P.network.items():
        #     #     assembly_slice = _vals['assembly_index']
        #     #     # self.W_mean_assembly[_assembly][dump] = np.mean(self.synapses['EE'][assembly_slice].w_clopath)
        #     #     _W = self.W_trained['EE'][assembly_slice, assembly_slice]
        #     #     self.W_in[_assembly][dump] = np.mean(_W[np.nonzero(_W)])
        # print(self.W_trained['EE']['W'][:50,:50])
        # file_name = 'W_assemblies'
        # self._save_data(data=self.W_assemblies, file_name=file_name, data_directory=data_directory)
        # file_name = 'W_trained'
        # self._save_data(data=self.W_trained, file_name=file_name, data_directory=data_directory)

        # end_timer = timer.time()
        # print('run duration:', str(end_timer-start_timer))

class exp_zenke_test_ff_plastic(Exp):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        super(exp_zenke_test_ff_plastic, self).__init__(**kwargs)

    def run(self, **kwargs):
        '''
        '''
        # directory and file name to store data
        #====================================================================
        group_data_directory = 'Datatemp/'+__name__+'/'
        group_data_filename = __name__+'_data.pkl'
        group_data_filename_train = __name__+'_data_train.pkl'
        group_data_filename_test = __name__+'_data_test.pkl'

        # load parameters
        #====================================================================
        # default. all parameter groups are initially called '1', e.g. P.neurons['1']
        P = ParamZenke2015()
        if 'neurons' in kwargs:
            if 'E_rate' in kwargs['neurons']:
                P.neurons['E_rate'].update(kwargs['neurons']['E_rate'])
        print(P.neurons['E_rate'])

        # free parameters
        # rates and weights of feedforward poisson inputs
        # timescale and set point of homeostatic plasticity
        # weights of IE synapses
        N_E = 4096
        N_I = 1024
        N_E_input=N_E
        p_connect = 0.1
        p_connect_input=0.05
        w_EE_init=0.1
        w_IE_init=0.15
        w_EI_init=0.2
        w_II_init=0.2
        w_FE_init=.35
        w_cons_init=0.
        reinit_W = False
        if 'reinit_W' in kwargs:
            reinit_W = kwargs['reinit_W']

        if 'w_FE_init' in kwargs:
            w_FE_init = kwargs['w_FE_init']
        if 'w_IE_init' in kwargs:
            w_IE_init = kwargs['w_IE_init']

        # initialize trained weight matrices from previous runs
        #-----------------------------------------------------
        data_directory = '_Data/'+self.experiment_name+'/'
        # specify filename or get most recent
        file_name_W = None
        if 'trial_id' in kwargs and kwargs['trial_id'] is not None:
            file_name_W = 'W_'+kwargs['trial_id']+'.pkl'
            file_name_W_cons = 'W_cons_'+kwargs['trial_id']+'.pkl'
        if file_name_W is None:
            files = os.listdir(data_directory)
            files_W = [file for file in files if 'W_2' in file]
            files_W_cons = [file for file in files if 'W_cons_2' in file]
            files_W.sort()
            files_W_cons.sort()
            file_name_W = files_W[-1]
            file_name_W_cons = files_W_cons[-1]
        # trial_id_load=''
        # file_name= 'W'+trial_id_load+'.pkl'
        if reinit_W:
            print(file_name_W)
            with open(data_directory+file_name_W, 'rb') as pkl_file:
                self.W_pretrained = pickle.load(pkl_file)
            with open(data_directory+file_name_W_cons, 'rb') as pkl_file:
                self.W_cons_pretrained = pickle.load(pkl_file)

        # load equations for adaptive exponential integrate and fire neuron
        #------------------------------------------------------------------
        Eq = equations.Zenke2015()

        neurons={}
        synapses={}

        # excitatory neurons
        #-------------------------------------------------------------------
        neuron_group='E'
        P.neurons[neuron_group]['N']=N_E
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_e, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams, 
            events={'consolidate':Eq.neuron_consolidation_event}
            )
        
        # inhibitory neurons
        #-----------------------------------------------------------
        neuron_group='I'
        P.neurons[neuron_group]['N']=N_I
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_i, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
            )

        # global online rate detector
        #---------------------------------------------------------------
        neuron_group='E_rate'
        P.neurons[neuron_group]['N']=1
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_global_rate, method='euler', name='neurons_'+neuron_group, namespace=nparams, 
            )
        # link global rate variable to all inhibitory neurons
        #---------------------------------------------------
        neurons['I'].H = linked_var(neurons['E_rate'], 'H')

        # connect rate detector to excitatory neurons
        #-----------------------------------------
        synapse_group='E_rate'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E_rate'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E_rate'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True'
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['E_rate'], Eq.synapse_global_rate, 
            on_pre=Eq.synapse_global_rate_pre, 
            namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler,)
        # connect synapses
        #--------------------
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])

        # recurrent EE synapses
        ###################################################################
        synapse_group='EE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'i!=j'
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['E'], Eq.synapse_e, 
            on_pre={'pre':Eq.synapse_e_pre,'consolidation':Eq.synapse_consolidate_on_event}, 
            on_event={'pre':'spike', 'consolidation':'consolidate'},
            on_post=Eq.synapse_e_post, 
            namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group], W_cons=self.W_cons_pretrained[synapse_group])
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],W_cons=self.W_cons_pretrained[synapse_group],key='w_exc_plastic')
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_cons_pretrained[synapse_group],W_cons=self.W_cons_pretrained[synapse_group],key='w_cons')
        else:
            # connect synapses
            #--------------------
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #-----------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)
            P.init_synapses[synapse_group]['w_cons'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_cons_init)

        # recurrent EI synapses
        #####################################################################
        synapse_group='EI'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.synapses[synapse_group]['w_exc_plastic_init']=w_EI_init
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # P.synapses[synapse_group]['A'] =  0. # turn plasticity off
        # P.synapses[synapse_group]['B'] =  0. # turn plasticity off
        # P.synapses[synapse_group]['Beta'] =  0. # turn plasticity off
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['I'], Eq.synapse_e_static, on_pre=Eq.synapse_e_pre_static, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            # synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_exc_plastic_init')
            # synapses[synapse_group].w_exc_plastic
        else:
            # connect synapses
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #-----------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_exc_plastic_init'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EI_init)
            # P.init_synapses[synapse_group]['w_exc_plastic_init'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EI_init)

        # recurrent IE synapses
        #####################################################################
        synapse_group='IE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['I'])
        P.synapses[synapse_group]['update_gaba_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['I'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['I'], neurons['E'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_inh_plastic')
        else:
            # connect synapses
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #---------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_inh_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_IE_init)

        # recurrent II synapses
        #####################################################################
        synapse_group='II'
        # copy default parameters
        #---------------------------
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['I'])
        P.synapses[synapse_group]['update_gaba_online']=0
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['I'])
        # connection conditions
        #-----------------------
        P.synapses[synapse_group]['connect_condition'] =  'i!=j' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        P.synapses[synapse_group]['eta'] =  0. # turn plasticity off
        # make synapses
        #------------------
        sparams = P.synapses[synapse_group]
        synapses[synapse_group] = Synapses(neurons['I'], neurons['I'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_inh_plastic')
        else:
            # connect synapses
            #------------------
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #---------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_inh_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_II_init)

        # design input images
        ####################################################################
        # n_states = 8
        # n_repeats = 30
        # n_transitions = n_states*n_repeats
        # # n_transitions = 8*30#8*10*5#10#8*10*5#8*10# 201 # number of image changes
        # transitions = np.zeros(n_transitions, dtype=int) 
        # transitions[1::8]=1 # circles
        # transitions[3::8]=2 # squares
        # transitions[5::8]=3 # crosses
        # transitions[7::8]=4 # plus
        # transitions[0::2]=0 # empty
        # stim_dur = 1000 # stimulus duration (ms)

        n_states = 12
        n_repeats = 12
        n_transitions = n_states*n_repeats#8*30#8*10*5#10#8*10*5#8*10# 201 # number of image changes
        transitions = np.zeros(n_transitions, dtype=int) 
        transitions[2::12]=1 # circles
        transitions[5::12]=2 # squares
        transitions[8::12]=3 # crosses
        transitions[11::12]=4 # plus
        transitions[0::3]=0 # empty
        transitions[1::3]=0 # empty
        transitions[:3*n_states]=0
        # print 'transitions',transitions
        stim_dur = 1000 # stimulus duration (ms)
        sim_duration = n_transitions*(stim_dur)*ms # total simulation duration
        # get list of images and onset times
        self.images, self.times, self.image_arrays = self._design_image_transitions(transitions=transitions, dt=stim_dur)
        # timed array for input rates based on images
        E_input_timed_array = self._image_to_timed_array(images=self.images, times=self.times, image_arrays=self.image_arrays, dt=stim_dur)
        # give input neurons acess to timed array
        P.neurons['E_input']['E_input_timed_array']=E_input_timed_array

        # input layer neurons
        ####################################################################
        neuron_group='E_input'
        P.neurons[neuron_group]['N']=N_E_input
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_poisson_traces, threshold=nparams['threshold_condition'], reset=Eq.neuron_poisson_traces_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams, 
            )

        # feedforward E synapses
        ###################################################################
        synapse_group='E_input_E'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.synapses[synapse_group]['update_ampa_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E_input'], neurons['E'], Eq.synapse_e, 
            on_pre=Eq.synapse_e_pre, 
            on_event={'post':'spike', 'consolidation':'consolidate'},
            on_post={'post':Eq.synapse_e_post,'consolidation':Eq.synapse_consolidate_on_event}, 
            namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])

        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],W_cons=self.W_cons_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_exc_plastic',W_cons=self.W_cons_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_cons_pretrained[synapse_group],key='w_cons',W_cons=self.W_cons_pretrained[synapse_group])
            self.sort_by_receptive_field_loc=self.W_pretrained['sort_by_receptive_field_loc']
        else:
            # create receptive fields and connect synapses
            #-------------------------------------------------
            _n_pixels = 64
            _radius = 8
            # get receptive field centers
            _center_x = np.array(_radius+1+(_n_pixels-2*_radius)*np.random.rand(N_E), dtype=int)
            _center_y = np.array(_radius+1+(_n_pixels-2*_radius)*np.random.rand(N_E), dtype=int) 
            centers = zip(_center_x, _center_y)
            # preallocate lists of pre and post indicdes
            i_list = [None]*N_E
            j_list = [None]*N_E
            # iterate over receptive field centers
            for center_i, center in enumerate(centers):
                # get index of all pixels in receptive field
                x,y,_i = self._get_receptive_field(center_x=center[0], center_y=center[1], radius=_radius, n_pixels=64)
                # corresponding list of post indices
                _j = [center_i]*len(_i)
                # add to master list
                i_list[center_i]=_i
                j_list[center_i]=_j
            # flatten master list of indices
            j_list = [item for sublist in j_list for item in sublist]
            i_list = [item for sublist in i_list for item in sublist]
            # connect synapses
            synapses[synapse_group].connect(i=i_list, j=j_list)
            # get raveled index of receptive field location for each recurrent neuron for sorting output data
            self.sort_by_receptive_field_loc = np.argsort(np.ravel_multi_index(multi_index=(_center_x, _center_y), dims=(_n_pixels,_n_pixels), order='F'))

            # initialize uniform weights 
            #-----------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FE_init)
            P.init_synapses[synapse_group]['w_cons'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_cons_init)

        # initial conditions
        #####################################################################
        self._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
        if not reinit_W:
            self._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

        # set up recording
        #####################################################################
        # recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
        # state monitors
        #-----------------
        # P.neurons['E_input']['rec_variables'].extend(['rates'])
        # P.neurons['E_input']['rec_indices'] = range(1000)
        # P.neurons['E_input']['rec_variables'].extend(['u_stp','x_stp'])
        # P.neurons['E_input']['rec_indices'] = range(10)
        # P.neurons['E']['rec_variables'].extend(['z_minus'])
        # P.neurons['E']['rec_indices'] = range(10)
        # P.neurons['I']['rec_variables'].extend(['G'])
        # P.neurons['I']['rec_indices'] = range(10)
        # P.synapses['EE']['rec_variables'].extend(['w_cons',])
        # P.synapses['EE']['rec_indices'] = synapses['EE'][slice(0,10),slice(0,10)]
        # P.synapses['E_input_E']['rec_variables'].extend(['w_exc_plastic','w_cons',])
        # P.synapses['E_input_E']['rec_indices'] = synapses['E_input_E'][slice(0,400),slice(0,400)]
        # P.synapses['IE']['rec_variables'].extend(['w_inh_plastic',])
        # P.synapses['IE']['rec_indices'] = synapses['IE'][slice(0,100),slice(0,100)]
        # P.neurons['E_rate']['rec_variables'].extend(['G'])
        self.rec = self._build_state_rec(brian_objects=[neurons, synapses,], keys=['neurons', 'synapses',], P=P)
        # spike monitors
        #--------------------- 
        spike_rec_N =  256
        P.simulation['1']['spike_rec_N']=spike_rec_N
        self.spike_rec_sort_by_receptive_field = self.sort_by_receptive_field_loc[:spike_rec_N]
        # P.simulation['1']['spike_rec_groups']=['E','I','E_input']
        # P.simulation['1']['spike_rec_groups']=[]
        # P.simulation['1']['spike_rec_groups']=[]
        # spike recording
        self.spike_rec = self._build_spike_rec(brian_objects=[neurons], keys=['neurons'], P=P)
        # self.spike_rec = SpikeMonitor(inputs['I'])
        # population rate monitors
        #--------------------------
        self.rate_rec={}
        # self.rate_rec['assembly']=PopulationRateMonitor(neurons['E'][slice(assembly_size)])
        # self.rate_rec['other']=PopulationRateMonitor(neurons['E'][assembly_size:2*assembly_size])

        # set up network
        #####################################################################
        self.net = Network()
        # make sure you collect 
        self.net = self._collect_brian_objects(self.net, neurons, synapses, self.rec['neurons'], self.rec['synapses'], self.spike_rec['neurons'], self.rate_rec)
        
        # self.net.add(object_container[group_key])
        # self.net.remove(*objs)
        # set up simulation parameters
        #####################################################################
        # set time step
        # prefs.codegen.target='numpy'
        # prefs.codegen.target='cython'

        defaultclock.dt = P.simulation['1']['dt']
        P.simulation['1']['run_time'] = sim_duration
        
        # apply field
        ####################################################################
        # field_timed_array = self._build_field_timed_array(field_on=10*ms, field_off=100*ms, field_mag=100*pA)

        # P.neurons['E']['I_field'] = 'field_timed_array(t)'
        # neurons['E'].I_field = P.neurons['E']['I_field']

        # run simulation
        ###################################################################
        # regular run
        #-------------
        # report progress every minute
        report = 'text'
        report_period = 60*second
        # self.net.run(P.simulation['1']['run_time'], report=report, report_period=report_period)

        # run in chunks and report
        #---------------------------
        n_burn_in = 2
        interval_duration = n_states*stim_dur*ms#sim_duration/n_intervals
        n_record=4096
        neurons['E_rate'].H=0.
        for _i in range(n_repeats):
            if _i<n_burn_in:
                synapses['EE'].namespace['A'] = 0.
                synapses['EE'].namespace['B'] = 0.
                synapses['EE'].namespace['Beta'] = 0.
                synapses['EE'].namespace['delta'] = 0.
                synapses['E_input_E'].namespace['A'] = 0.
                synapses['E_input_E'].namespace['B'] = 0.
                synapses['E_input_E'].namespace['Beta'] = 0.
                synapses['E_input_E'].namespace['delta'] = 0.
                synapses['IE'].namespace['eta'] = 0.
            elif _i==n_burn_in:
                synapses['EE'].namespace['A'] = P.synapses['E']['A']
                synapses['EE'].namespace['B'] = P.synapses['E']['B']
                synapses['EE'].namespace['Beta'] = P.synapses['E']['Beta']
                synapses['EE'].namespace['delta'] = P.synapses['E']['delta']
                synapses['E_input_E'].namespace['A'] = P.synapses['E']['A']
                synapses['E_input_E'].namespace['B'] = P.synapses['E']['B']
                synapses['E_input_E'].namespace['Beta'] = P.synapses['E_input_E']['Beta']
                synapses['E_input_E'].namespace['delta'] = P.synapses['E_input_E']['delta']
                synapses['IE'].namespace['eta'] = P.synapses['I']['eta']
            _spike_mon = SpikeMonitor(neurons['E'][:n_record])
            _spike_mon_2 = SpikeMonitor(neurons['E_input'][:n_record])
            self.net.add(_spike_mon)
            self.net.add(_spike_mon_2)
            self.net.run(interval_duration, report=report, report_period=report_period)
            test= self._spike_monitor_to_binary_array(spike_monitor=_spike_mon, run_time=interval_duration,t_0=_i*interval_duration)
            test_2= self._spike_monitor_to_binary_array(spike_monitor=_spike_mon_2, run_time=interval_duration,t_0=_i*interval_duration)
            self.net.remove(_spike_mon)
            self.net.remove(_spike_mon_2)
            test = np.sum(test[:,:].reshape(test[:,:].shape[0],-1, 10), axis=2)
            test_2 = np.sum(test_2[:,:].reshape(test_2[:,:].shape[0],-1, 10), axis=2)
            plt.figure()
            plt.imshow(test[:,:], cmap='binary',aspect='auto')
            plt.show(block=False)
            plt.figure()
            plt.imshow(test_2[:,:], cmap='binary',aspect='auto')
            plt.show(block=False)

            # print mean weights0
            print('EE',np.mean(synapses['EE'].w_exc_plastic))
            print('E_input_E',np.mean(synapses['E_input_E'].w_exc_plastic))
            print('IE', np.mean(synapses['IE'].w_inh_plastic))
            print('global excitatory rate:', neurons['E_rate'].G)
            print('run ', str(_i+1),' of ', str(n_repeats), ' completed')

            del _spike_mon
            del test

    
        # FIXME set as attributes
        #-------------------------
        self.P = P
        # self.inputs=inputs
        self.synapses=synapses
        self.neurons=neurons

        # save weight matrices
        data_directory = '_Data/'+self.experiment_name+'/'
        if 'trial_id' in kwargs and kwargs['trial_id'] is not None:
            trial_id=kwargs['trial_id']
        else:
            trial_id = self._generate_trial_id()
            keyword='_'.join(['t_H', str(P.neurons['E_rate']['t_H']), 'gamma', str(P.neurons['E_rate']['t_H']), 'IE', str(w_IE_init), 'FE', str(w_FE_init) ])
            trial_id=trial_id+keyword
        file_name_W = 'W'+'_'+trial_id
        file_name_W_cons = 'W_cons'+'_'+trial_id
        self.W = {}
        self.W_cons = {}
        #-------------------------------------------------
        self.W['E_input_E'] = self._to_weight_matrix(synapse_group=self.synapses['E_input_E'], w_key='w_exc_plastic')
        self.W_cons['E_input_E'] = self._to_weight_matrix(synapse_group=self.synapses['E_input_E'], w_key='w_cons')
        self.W['EE'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_exc_plastic')
        self.W_cons['EE'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_cons')
        self.W['IE'] = self._to_weight_matrix(synapse_group=self.synapses['IE'], w_key='w_inh_plastic')
        self.W['II'] = self._to_weight_matrix(synapse_group=self.synapses['II'], w_key='w_inh_plastic')
        self.W['EI'] = self._to_weight_matrix(synapse_group=self.synapses['EI'], w_key='w_exc_plastic_init')
        self.W['sort_by_receptive_field_loc'] = self.sort_by_receptive_field_loc

        self._save_data(data=self.W, file_name=file_name_W, data_directory=data_directory)
        self._save_data(data=self.W_cons, file_name=file_name_W_cons, data_directory=data_directory)

        # self.timed_rates = timed_rates

        # dump weights
        #------------------------------------------------------------
        # data directories and file names
        #-------------------------------------
        # data_directory = '_Data/'+self.experiment_name+'/'
        # n_dumps = n_episodes
        # run_time_dump = P.simulation['1']['run_time']/n_dumps
        # # preallocate array for mean 
        # W_out_pairs = itertools.product(P.network.keys(), repeat=2)
        # if reinit_W:
        #     for _pair in W_pretrained_assemblies:
        #         W_pretrained_assemblies[_pair]['mean'] = np.append(W_pretrained_assemblies[_pair]['mean'], np.zeros(n_dumps))
        #         W_pretrained_assemblies[_pair]['std'] = np.append(W_pretrained_assemblies[_pair]['std'], np.zeros(n_dumps))

        #     self.W_assemblies = W_pretrained_assemblies
        # else:
        #     self.W_assemblies={}
        #     for _pair in W_out_pairs:
        #         _assemblies = '_'.join(_pair)
        #         self.W_assemblies[_assemblies] = {
        #                 'mean':np.zeros(n_dumps),
        #                 'std':np.zeros(n_dumps),}
        # #     sl
        # self.W_trained={'EE':{}, 'IE':{}}
        # if reinit_W:
        #     self.W_trained['EE']['W_old'] = W_pretrained['EE']['W']
        #     self.W_trained['IE']['W_old'] = W_pretrained['IE']['W']

        # start_timer = timer.time()
        # for dump in range(n_dumps):
        #     # run
        #     #--------------------------
        #     self.net.run(run_time_dump)

        #     # get weight matrix 
        #     #----------------------
        #     self.W_trained['EE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_clopath',)
        #     self.W_trained['IE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['IE'], w_key='w_vogels',)

        #     self.W_trained['EE']['i'] = list(self.synapses['EE'].i)
        #     self.W_trained['EE']['j'] = list(self.synapses['EE'].j)
        #     self.W_trained['IE']['i'] = list(self.synapses['IE'].i)
        #     self.W_trained['IE']['j'] = list(self.synapses['IE'].j)

        #     # iterate over assembly combinations
        #     #-------------------------------------
        #     print('assemblies',self.W_assemblies.keys())
        #     for _assemblies in self.W_assemblies.keys():

        #         _pre = _assemblies.split('_')[0]
        #         _post = _assemblies.split('_')[1]
        #         _pre_slice = P.network[_pre]['assembly_index']
        #         _post_slice = P.network[_post]['assembly_index']
        #         _W = self.W_trained['EE']['W'][_pre_slice, _post_slice]
        #         self.W_assemblies[_assemblies]['mean'][dump-n_dumps] = np.mean(_W[np.nonzero(_W)])
        #         self.W_assemblies[_assemblies]['std'][dump-n_dumps] = np.std(_W[np.nonzero(_W)])

        #     # iterate over assemblies and store mean weights
        #     #------------------------------------------------
        #     # for _assembly, _vals in P.network.items():
        #     #     assembly_slice = _vals['assembly_index']
        #     #     # self.W_mean_assembly[_assembly][dump] = np.mean(self.synapses['EE'][assembly_slice].w_clopath)
        #     #     _W = self.W_trained['EE'][assembly_slice, assembly_slice]
        #     #     self.W_in[_assembly][dump] = np.mean(_W[np.nonzero(_W)])
        # print(self.W_trained['EE']['W'][:50,:50])
        # file_name = 'W_assemblies'
        # self._save_data(data=self.W_assemblies, file_name=file_name, data_directory=data_directory)
        # file_name = 'W_trained'
        # self._save_data(data=self.W_trained, file_name=file_name, data_directory=data_directory)

        # end_timer = timer.time()
        # print('run duration:', str(end_timer-start_timer))


class exp_zenke_test_ff_plastic_temp(Exp):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        super(exp_zenke_test_ff_plastic_temp, self).__init__(**kwargs)

    def run(self, **kwargs):
        '''
        '''
        # directory and file name to store data
        #====================================================================
        group_data_directory = 'Datatemp/'+__name__+'/'
        group_data_filename = __name__+'_data.pkl'
        group_data_filename_train = __name__+'_data_train.pkl'
        group_data_filename_test = __name__+'_data_test.pkl'

        # load parameters
        #====================================================================
        # default. all parameter groups are initially called '1', e.g. P.neurons['1']
        P = ParamZenke2015()

        # free parameters
        # rates and weights of feedforward poisson inputs
        # timescale and set point of homeostatic plasticity
        # weights of IE synapses
        N_E = 4096
        N_I = 1024
        N_E_input=N_E
        p_connect = 0.1
        p_connect_input=0.05
        w_EE_init=0.1
        w_IE_init=0.15
        w_EI_init=0.2
        w_II_init=0.2
        w_FE_init=0.5
        reinit_W = True

        # initialize trained weight matrices from previous runs
        #-----------------------------------------------------
        data_directory = 'Data/'+self.experiment_name+'/'
        # specify filename or get most recent
        file_name_W = None
        if file_name_W is None:
            files = os.listdir(data_directory)
            files_W = [file for file in files if 'W_2' in file]
            files_W_cons = [file for file in files if 'W_cons_2' in file]
            files_W.sort()
            files_W_cons.sort()
            file_name_W = files_W[-1]
            file_name_W_cons = files_W_cons[-1]
        # trial_id_load=''
        # file_name= 'W'+trial_id_load+'.pkl'
        if reinit_W:
            with open(data_directory+file_name_W, 'rb') as pkl_file:
                self.W_pretrained = pickle.load(pkl_file)
            with open(data_directory+file_name_W_cons, 'rb') as pkl_file:
                self.W_cons_pretrained = pickle.load(pkl_file)

        # load equations for adaptive exponential integrate and fire neuron
        #------------------------------------------------------------------
        Eq = equations.Zenke2015()

        neurons={}
        synapses={}

        # excitatory neurons
        #-------------------------------------------------------------------
        neuron_group='E'
        P.neurons[neuron_group]['N']=N_E
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_e, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams, 
            events={'consolidate':Eq.neuron_consolidation_event}
            )
        
        # inhibitory neurons
        #-----------------------------------------------------------
        neuron_group='I'
        P.neurons[neuron_group]['N']=N_I
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_i, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
            )

        # global online rate detector
        #---------------------------------------------------------------
        neuron_group='E_rate'
        P.neurons[neuron_group]['N']=1
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_global_rate, method='euler', name='neurons_'+neuron_group, namespace=nparams, 
            )
        # link global rate variable to all inhibitory neurons
        #---------------------------------------------------
        neurons['I'].H = linked_var(neurons['E_rate'], 'H')

        # connect rate detector to excitatory neurons
        #-----------------------------------------
        synapse_group='E_rate'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E_rate'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E_rate'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True'
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['E_rate'], Eq.synapse_global_rate, 
            on_pre=Eq.synapse_global_rate_pre, 
            namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler,)
        # connect synapses
        #--------------------
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])

        # recurrent EE synapses
        ###################################################################
        synapse_group='EE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'i!=j'
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['E'], Eq.synapse_e, 
            on_pre={'pre':Eq.synapse_e_pre,'consolidation':Eq.synapse_consolidate_on_event}, 
            on_event={'pre':'spike', 'consolidation':'consolidate'},
            on_post=Eq.synapse_e_post, 
            namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_exc_plastic')
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_cons_pretrained[synapse_group],key='w_cons')
        else:
            # connect synapses
            #--------------------
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #-----------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)
            P.init_synapses[synapse_group]['w_cons'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)

        # recurrent EI synapses
        #####################################################################
        synapse_group='EI'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.synapses[synapse_group]['w_exc_plastic_init']=w_EI_init
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # P.synapses[synapse_group]['A'] =  0. # turn plasticity off
        # P.synapses[synapse_group]['B'] =  0. # turn plasticity off
        # P.synapses[synapse_group]['Beta'] =  0. # turn plasticity off
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['I'], Eq.synapse_e_static, on_pre=Eq.synapse_e_pre_static, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            # synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_exc_plastic_init')
            # synapses[synapse_group].w_exc_plastic
        else:
            # connect synapses
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #-----------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_exc_plastic_init'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EI_init)
            # P.init_synapses[synapse_group]['w_exc_plastic_init'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_EI_init)

        # recurrent IE synapses
        #####################################################################
        synapse_group='IE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['I'])
        P.synapses[synapse_group]['update_gaba_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['I'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['I'], neurons['E'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_inh_plastic')
        else:
            # connect synapses
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #---------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_inh_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_IE_init)

        # recurrent II synapses
        #####################################################################
        synapse_group='II'
        # copy default parameters
        #---------------------------
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['I'])
        P.synapses[synapse_group]['update_gaba_online']=0
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['I'])
        # connection conditions
        #-----------------------
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  p_connect
        P.synapses[synapse_group]['eta'] =  0. # turn plasticity off
        # make synapses
        #------------------
        sparams = P.synapses[synapse_group]
        synapses[synapse_group] = Synapses(neurons['I'], neurons['I'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])
        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_inh_plastic')
        else:
            # connect synapses
            #------------------
            synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])
            # initialize uniform weights 
            #---------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_inh_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_II_init)

        # design input images
        ####################################################################
        n_transitions = 8#8*10*5#8*10# 201 # number of image changes
        transitions = np.zeros(n_transitions, dtype=int) 
        transitions[1::8]=1 # circles
        transitions[3::8]=2 # squares
        transitions[5::8]=3 # crosses
        transitions[7::8]=4 # plus
        transitions[0::2]=0 # empty
        stim_dur = 50 # stimulus duration (ms)
        sim_duration = n_transitions*(stim_dur)*ms # total simulation duration
        # get list of images and onset times
        self.images, self.times, self.image_arrays = self._design_image_transitions(transitions=transitions, dt=stim_dur)
        # timed array for input rates based on images
        E_input_timed_array = self._image_to_timed_array(images=self.images, times=self.times, image_arrays=self.image_arrays, dt=stim_dur)
        # give input neurons acess to timed array
        P.neurons['E_input']['E_input_timed_array']=E_input_timed_array

        # input layer neurons
        ####################################################################
        neuron_group='E_input'
        P.neurons[neuron_group]['N']=N_E_input
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron_poisson_traces, threshold=nparams['threshold_condition'], reset=Eq.neuron_poisson_traces_reset,   refractory=nparams['refractory'],  method='euler', name='neurons_'+neuron_group, namespace=nparams, 
            )

        # feedforward E synapses
        ###################################################################
        synapse_group='E_input_E'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['E'])
        P.synapses[synapse_group]['update_ampa_online']=1
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['E'])
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E_input'], neurons['E'], Eq.synapse_e, 
            on_pre=Eq.synapse_e_pre, 
            on_event={'post':'spike', 'consolidation':'consolidate'},
            on_post={'post':Eq.synapse_e_post,'consolidation':Eq.synapse_consolidate_on_event}, 
            namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler, delay=P.synapses[synapse_group]['delay'])

        # initialize weights from saved data
        #------------------------------------
        if reinit_W:
            synapses[synapse_group] = self._connect_synapses_from_weight_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],)
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_pretrained[synapse_group],key='w_exc_plastic')
            synapses[synapse_group] = self._initialize_weights_from_matrix(synapses=synapses[synapse_group], W=self.W_cons_pretrained[synapse_group],key='w_cons')
            self.sort_by_receptive_field_loc=self.W_pretrained['sort_by_receptive_field_loc']
        else:
            # create receptive fields and connect synapses
            #-------------------------------------------------
            _n_pixels = 64
            _radius = 8
            # get receptive field centers
            _center_x = np.array(_radius+1+(_n_pixels-2*_radius)*np.random.rand(N_E), dtype=int)
            _center_y = np.array(_radius+1+(_n_pixels-2*_radius)*np.random.rand(N_E), dtype=int) 
            centers = zip(_center_x, _center_y)
            # preallocate lists of pre and post indicdes
            i_list = [None]*N_E
            j_list = [None]*N_E
            # iterate over receptive field centers
            for center_i, center in enumerate(centers):
                # get index of all pixels in receptive field
                x,y,_i = self._get_receptive_field(center_x=center[0], center_y=center[1], radius=_radius, n_pixels=64)
                # corresponding list of post indices
                _j = [center_i]*len(_i)
                # add to master list
                i_list[center_i]=_i
                j_list[center_i]=_j
            # flatten master list of indices
            j_list = [item for sublist in j_list for item in sublist]
            i_list = [item for sublist in i_list for item in sublist]
            # connect synapses
            synapses[synapse_group].connect(i=i_list, j=j_list)
            # get raveled index of receptive field location for each recurrent neuron for sorting output data
            self.sort_by_receptive_field_loc = np.argsort(np.ravel_multi_index(multi_index=(_center_x, _center_y), dims=(_n_pixels,_n_pixels), order='F'))

            # initialize uniform weights 
            #-----------------------------
            # number of synapses
            Nsyn = len(synapses[synapse_group].i)
            # initial weight matrices
            P.init_synapses[synapse_group]['w_exc_plastic'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FE_init)
            P.init_synapses[synapse_group]['w_cons'] = P._weight_array_uniform(Nsyn=Nsyn, w=w_FE_init)

        # initial conditions
        #####################################################################
        self._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
        if not reinit_W:
            self._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

        # set up recording
        #####################################################################
        # recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
        # state monitors
        #-----------------
        # P.neurons['E_input']['rec_variables'].extend(['rates'])
        # P.neurons['E_input']['rec_indices'] = range(1000)
        # P.neurons['E_input']['rec_variables'].extend(['u_stp','x_stp'])
        # P.neurons['E_input']['rec_indices'] = range(10)
        # P.neurons['E']['rec_variables'].extend(['u_stp','x_stp'])
        # P.neurons['E']['rec_indices'] = range(10)
        # P.neurons['I']['rec_variables'].extend(['G'])
        # P.neurons['I']['rec_indices'] = range(10)
        # P.synapses['EE']['rec_variables'].extend(['w_cons',])
        # P.synapses['EE']['rec_indices'] = synapses['EE'][slice(0,10),slice(0,10)]
        # P.synapses['E_input_E']['rec_variables'].extend(['w_exc_plastic','w_cons',])
        # P.synapses['E_input_E']['rec_indices'] = synapses['E_input_E'][slice(0,200),slice(0,200)]
        # P.synapses['IE']['rec_variables'].extend(['w_inh_plastic',])
        # P.synapses['IE']['rec_indices'] = synapses['IE'][slice(0,100),slice(0,100)]
        self.rec = self._build_state_rec(brian_objects=[neurons, synapses,], keys=['neurons', 'synapses',], P=P)
        # spike monitors
        #--------------------- 
        spike_rec_N =  256
        P.simulation['1']['spike_rec_N']=spike_rec_N
        self.spike_rec_sort_by_receptive_field = self.sort_by_receptive_field_loc[:spike_rec_N]
        # P.simulation['1']['spike_rec_groups']=['E']
        P.simulation['1']['spike_rec_groups']=[]
        # spike recording
        self.spike_rec = self._build_spike_rec(brian_objects=[neurons], keys=['neurons'], P=P)
        # self.spike_rec = SpikeMonitor(inputs['I'])
        # population rate monitors
        #--------------------------
        self.rate_rec={}
        # self.rate_rec['assembly']=PopulationRateMonitor(neurons['E'][slice(assembly_size)])
        # self.rate_rec['other']=PopulationRateMonitor(neurons['E'][assembly_size:2*assembly_size])

        # set up network
        #####################################################################
        self.net = Network()
        # make sure you collect 
        self.net = self._collect_brian_objects(self.net, neurons, synapses, self.rec['neurons'], self.rec['synapses'], self.spike_rec['neurons'], self.rate_rec)
        
        # self.net.add(object_container[group_key])
        # self.net.remove(*objs)
        # set up simulation parameters
        #####################################################################
        # set time step
        # prefs.codegen.target='numpy'
        # prefs.codegen.target='cython'

        defaultclock.dt = P.simulation['1']['dt']
        P.simulation['1']['run_time'] = sim_duration
        
        # apply field
        ####################################################################
        # field_timed_array = self._build_field_timed_array(field_on=10*ms, field_off=100*ms, field_mag=100*pA)

        # P.neurons['E']['I_field'] = 'field_timed_array(t)'
        # neurons['E'].I_field = P.neurons['E']['I_field']

        # run simulation
        ###################################################################
        # regular run
        #-------------
        # report progress every minute
        report = 'text'
        report_period = 60*second
        self.net.run(P.simulation['1']['run_time'], report=report, report_period=report_period)
    
        # FIXME set as attributes
        #-------------------------
        self.P = P
        # self.inputs=inputs
        self.synapses=synapses
        self.neurons=neurons

        # save weight matrices
        data_directory = 'Data/'+self.experiment_name+'/'
        trial_id = self._generate_trial_id()
        file_name_W = 'W'+'_'+trial_id
        file_name_W_cons = 'W_cons'+'_'+trial_id
        self.W = {}
        self.W_cons = {}
        #-------------------------------------------------
        self.W['E_input_E'] = self._to_weight_matrix(synapse_group=self.synapses['E_input_E'], w_key='w_exc_plastic')
        self.W_cons['E_input_E'] = self._to_weight_matrix(synapse_group=self.synapses['E_input_E'], w_key='w_cons')
        self.W['EE'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_exc_plastic')
        self.W_cons['EE'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_cons')
        self.W['IE'] = self._to_weight_matrix(synapse_group=self.synapses['IE'], w_key='w_inh_plastic')
        self.W['II'] = self._to_weight_matrix(synapse_group=self.synapses['II'], w_key='w_inh_plastic')
        self.W['EI'] = self._to_weight_matrix(synapse_group=self.synapses['EI'], w_key='w_exc_plastic_init')
        self.W['sort_by_receptive_field_loc'] = self.sort_by_receptive_field_loc

        self._save_data(data=self.W, file_name=file_name_W, data_directory=data_directory)
        self._save_data(data=self.W_cons, file_name=file_name_W_cons, data_directory=data_directory)

        # self.timed_rates = timed_rates

        # dump weights
        #------------------------------------------------------------
        # data directories and file names
        #-------------------------------------
        # data_directory = '_Data/'+self.experiment_name+'/'
        # n_dumps = n_episodes
        # run_time_dump = P.simulation['1']['run_time']/n_dumps
        # # preallocate array for mean 
        # W_out_pairs = itertools.product(P.network.keys(), repeat=2)
        # if reinit_W:
        #     for _pair in W_pretrained_assemblies:
        #         W_pretrained_assemblies[_pair]['mean'] = np.append(W_pretrained_assemblies[_pair]['mean'], np.zeros(n_dumps))
        #         W_pretrained_assemblies[_pair]['std'] = np.append(W_pretrained_assemblies[_pair]['std'], np.zeros(n_dumps))

        #     self.W_assemblies = W_pretrained_assemblies
        # else:
        #     self.W_assemblies={}
        #     for _pair in W_out_pairs:
        #         _assemblies = '_'.join(_pair)
        #         self.W_assemblies[_assemblies] = {
        #                 'mean':np.zeros(n_dumps),
        #                 'std':np.zeros(n_dumps),}
        # #     sl
        # self.W_trained={'EE':{}, 'IE':{}}
        # if reinit_W:
        #     self.W_trained['EE']['W_old'] = W_pretrained['EE']['W']
        #     self.W_trained['IE']['W_old'] = W_pretrained['IE']['W']

        # start_timer = timer.time()
        # for dump in range(n_dumps):
        #     # run
        #     #--------------------------
        #     self.net.run(run_time_dump)

        #     # get weight matrix 
        #     #----------------------
        #     self.W_trained['EE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['EE'], w_key='w_clopath',)
        #     self.W_trained['IE']['W'] = self._to_weight_matrix(synapse_group=self.synapses['IE'], w_key='w_vogels',)

        #     self.W_trained['EE']['i'] = list(self.synapses['EE'].i)
        #     self.W_trained['EE']['j'] = list(self.synapses['EE'].j)
        #     self.W_trained['IE']['i'] = list(self.synapses['IE'].i)
        #     self.W_trained['IE']['j'] = list(self.synapses['IE'].j)

        #     # iterate over assembly combinations
        #     #-------------------------------------
        #     print('assemblies',self.W_assemblies.keys())
        #     for _assemblies in self.W_assemblies.keys():

        #         _pre = _assemblies.split('_')[0]
        #         _post = _assemblies.split('_')[1]
        #         _pre_slice = P.network[_pre]['assembly_index']
        #         _post_slice = P.network[_post]['assembly_index']
        #         _W = self.W_trained['EE']['W'][_pre_slice, _post_slice]
        #         self.W_assemblies[_assemblies]['mean'][dump-n_dumps] = np.mean(_W[np.nonzero(_W)])
        #         self.W_assemblies[_assemblies]['std'][dump-n_dumps] = np.std(_W[np.nonzero(_W)])

        #     # iterate over assemblies and store mean weights
        #     #------------------------------------------------
        #     # for _assembly, _vals in P.network.items():
        #     #     assembly_slice = _vals['assembly_index']
        #     #     # self.W_mean_assembly[_assembly][dump] = np.mean(self.synapses['EE'][assembly_slice].w_clopath)
        #     #     _W = self.W_trained['EE'][assembly_slice, assembly_slice]
        #     #     self.W_in[_assembly][dump] = np.mean(_W[np.nonzero(_W)])
        # print(self.W_trained['EE']['W'][:50,:50])
        # file_name = 'W_assemblies'
        # self._save_data(data=self.W_assemblies, file_name=file_name, data_directory=data_directory)
        # file_name = 'W_trained'
        # self._save_data(data=self.W_trained, file_name=file_name, data_directory=data_directory)

        # end_timer = timer.time()
        # print('run duration:', str(end_timer-start_timer))


class exp_litwinkumar_test(Exp):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        super(exp_litwinkumar_test, self).__init__(**kwargs)

    def run(self, **kwargs):
        '''
        '''
        # parameters
        #--------------
        P = ParamLitwinKumar2014()
        eq = equations.AdexLitwinKumar2014()

        # network size
        #---------------
        self.N_E = 100
        self.N_I = 25
        self.N_total = self.N_E+self.N_I

        # feedforward inputs
        #---------------------
        self.inputs_E = PoissonGroup(self.N_E, 2000.*Hz)
        self.inputs_I = PoissonGroup(self.N_I, 1000.*Hz)

        # for storing different neuron groups
        self.neurons={}
        self.synapses={}

        # excitatory neurons
        #--------------------------------
        neuron_group='E'
        self.neurons[neuron_group] =  NeuronGroup(self.N_E, eq.neuron, threshold='u>20*mV', reset=eq.neuron_reset,   refractory=1*ms,  method='euler', name='neurons_'+neuron_group, namespace=P.neurons['1']
            )

        # inhibitory neurons
        #--------------------------------
        neuron_group='I'
        self.neurons[neuron_group] =  NeuronGroup(self.N_I, eq.neuron, threshold='u>20*mV', reset=eq.neuron_reset,   refractory=1*ms,  method='euler', name='neurons_'+neuron_group, namespace=P.neurons['1']
            )


