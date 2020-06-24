from brian2 import *
import equations
import copy

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
        }

        self.neurons['1'] = {
            'rec_variables':['u'],
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
            # 'threshold_condition':'u>V_T+20*mV',
            'threshold_condition':'u2>=u_max',
            'I_after' : 400*pA,
            'a_adapt' : 4*nS,
            'b_adapt' : 0.805*pA,
            't_w_adapt' : 144*ms,
            't_z_after' : 40*ms,
            'u_reset' : -70*mV,
            'u_max':20*mV,

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

        self.synapses['1'] = {
            'rec_variables':[],
            'update_ampa_online':0,
            'update_nmda_online':0,   

            # synapse parameters
            #====================================================================
            # ampa
            #''''''''''''''''''''

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
            'include_homeostatic':0,

            # vogels
            #---------------------
            'tau_vogels':20*ms,
            'eta_vogels':.00001,
            'alpha_vogels':120,
            'w_max_vogels':10., 

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
            'N_E' : 100, 
            'N_I': 25,
            'syn_condition': 'i==1',
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

class Exp(object):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        pass

    def _build_spike_rec(self, brian_objects, keys, P):
        '''
        '''
        # zipped_objects = zip(keys, brian_objects)
        rec={}
        for i, key in enumerate(keys):
            rec[key] = {}
            for group_key, group in brian_objects[i].iteritems():
                brian_object = group

                rec[key][group_key]

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
            for group_key, group in brian_objects[i].iteritems():
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
            for group_key, group in brian_object.iteritems():
                for param, val in init_dic[group_key].iteritems():
                    if hasattr(brian_object[group_key], param):
                        setattr(brian_object[group_key], param, val)

        else:
            for param, val in init_dic[group_key].iteritems():
                if hasattr(brian_object, param):
                    setattr(brian_object, param, val)

    def _collect_brian_objects(self, net, *dics):
        '''
        '''
        for object_container in dics:
            for group_key, group in object_container.iteritems():
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
        for group_type_key, group_type in rec.iteritems():
            group_dict[group_type_key] = {}
            # iterate over groups
            for group_key, group in group_type.iteritems():

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
        for group_key, group in mon.iteritems():
            for var in group.record_variables:
                data = getattr(mon, var)
                index = group.record
                brian_group = group.source.name
                trial_id = P.simulation['trial_id']


        df = pd.DataFrame()
        for group_key, group in mon.iteritems():
            for var in group.record_variables:
                if var not in df:
                    df[var] = [getattr(mon, var)]
                df['brian_group'] = mon.source.name

                df['index'] = mon.record

                df['trial_id']

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
        N_E = 20
        N_I = 5
        w_EE_init=2.8
        w_IE_init=49.
        w_EI_init=1.3
        w_II_init=16.2
        w_FE_init=2.8
        w_FI_init=2.8
        N_assembly=3
        N_recall=1
        inputs={}


        # load equations for adaptive exponential integrate and fire neuron
        #------------------------------------------------------------------
        Eq = equations.AdexLitwinKumar2014()

        # design feedforward input
        #-------------------------
        
        inputs_I = PoissonGroup(N_I, 1000.*Hz)

        neurons={}
        synapses={}

        # excitatory neurons
        #-------------------------------------------------------------------
        neuron_group='E'
        P.neurons[neuron_group]=copy.deepcopy(P.neurons['1'])
        P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
        P.neurons[neuron_group]['N']=N_E
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
            )

        # inhibitory neurons
        #-----------------------------------------------------------
        neuron_group='I'
        P.neurons[neuron_group]=copy.deepcopy(P.neurons['1'])
        P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
        P.neurons[neuron_group]['N']=N_I
        nparams=P.neurons[neuron_group]
        neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
            )

        # parameter updates to be distributed to all synapses
        #----------------------------------------------------------
        P.synapses['1']['update_ampa_online']=0
        P.synapses['1']['update_gaba_online']=0
        P.synapses['1']['update_nmda_online']=0

        # recurrent EE synapses
        ###################################################################
        synapse_group='EE'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'i!=j'
        P.synapses[synapse_group]['connect_p'] =  0.2
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['E'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
        # connect synapses
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], p= P.synapses[synapse_group]['connect_p'])

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
        P.synapses[synapse_group]['connect_p'] =  0.4 
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['E'], neurons['I'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
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
        P.synapses[synapse_group]['update_gaba_online']=0
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] =  'True' 
        P.synapses[synapse_group]['connect_p'] =  0.4 
        # synapse parameters
        sparams = P.synapses[synapse_group]
        # make synapses
        synapses[synapse_group] = Synapses(neurons['I'], neurons['E'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
        # connect synapses
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
        P.synapses[synapse_group]['connect_p'] =  0.4 
        # make synapses
        #------------------
        sparams = P.synapses[synapse_group]
        synapses[synapse_group] = Synapses(neurons['I'], neurons['I'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
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
        inputs['E'] = PoissonGroup(N_E, 2000.*Hz)
        # recurrent EE synapses
        synapse_group='FE_train'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] = 'i==j'
        # synapse parameters
        
        # create synapses
        synapses[synapse_group] = Synapses(inputs['E'], neurons['E'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
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
        inputs['I']= PoissonGroup(N_I, 1000.*Hz)
        # recurrent EE synapses
        synapse_group='FI_train'
        # copy default parameters
        P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        # connect all to all
        P.synapses[synapse_group]['connect_condition'] = 'i==j'
        # synapse parameters
        
        # create synapses
        synapses[synapse_group] = Synapses(inputs['I'], neurons['I'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
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
        # P.init_synapses['2']=copy.deepcopy(P.init_synapses['1'])
        self._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
        self._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

        # set up recording
        #####################################################################
        # recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
        P.neurons['E']['rec_variables'].append('I_field')
        self.rec = self._build_state_rec(brian_objects=[neurons, synapses,], keys=['neurons', 'synapses',], P=P)

        # set up network
        #####################################################################
        net = Network()
        net = self._collect_brian_objects(net, inputs, neurons, synapses, self.rec['neurons'], self.rec['synapses'])

        # run simulation
        #####################################################################
        # set time step
        defaultclock.dt = P.simulation['1']['dt']
        P.simulation['1']['run_time']=10*ms
        self.net = net
        # net.run(P.simulation['1']['run_time'])
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


