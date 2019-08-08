'''
simulate network of 10 E and 3 I neurons as in clopath 2010.
10 feedforward poisson inputs, each projects to an assembly of 3 E and 1 I neurons, with each E neuron participating in at least 2 assemblies.
Feedforward inputs are activated randomly activated in succession at high rate (100 Hz) for 100 ms with weights updated online. dcs is applied in conjunction with one of the assemblies only.

to test recall, each neuron in the recurrent network is reactivated alone for 100 ms.  recall of the original assemblies is tested by overlap of firing rate during test period.  does dcs bias recall towards the assembly that it is paired with? 


'''
from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy
import equations
import run_control
import param
import inputs
import pandas as pd
import uuid
import analysis
import itertools

# FIXME use timed array to turn field on and off at specific time
# choose a poissoninput group that is always paired with dcs
# use the same timedarray (corresponding row from input_timed_array)

# FIXME assign spatial variable to each neuron and use that to design connectivity
# directory and file name to store data
#====================================================================
group_data_directory = 'Datatemp/'+__name__+'/'
group_data_filename = __name__+'_data.pkl'
group_data_filename_train = __name__+'_data_train.pkl'
group_data_filename_test = __name__+'_data_test.pkl'

# load parameters
#====================================================================
# default. all parameter groups are initially called '1', e.g. P.neurons['1']
P = param.Default()
# Clopath et al. 2010 parameters
Pclopath=  param.Clopath2010()

# use parameters directly from clopath model
P.synapses['1'] = Pclopath.synapses['1']
P.neurons['1'] = Pclopath.neurons['1']
P.synapses['1']['g_max_ampa']=.6*P.synapses['1']['g_max_ampa']
P.synapses['1']['g_max_nmda']=0.*P.synapses['1']['g_max_nmda']
P.synapses['1']['A_LTP']=200.*P.synapses['1']['A_LTP']
P.synapses['1']['A_LTD']=200.*P.synapses['1']['A_LTD']
P.synapses['1']['w_max_clopath']=4

# free parameters
# rates and weights of feedforward poisson inputs
# timescale and set point of homeostatic plasticity
# weights of IE synapses
N_E = 20
N_I = 5
w_EE_init=0.1
N_assembly=3
N_recall=1


# load equations for adaptive exponential integrate and fire neuron
#=====================================================================
Eq = equations.AdexBonoClopath()
#=====================================================================

# design feedforward input
#=======================================================================
# set up feedforward inputs 
# see :https://brian2.readthedocs.io/en/stable/user/input.html#setting-rates-for-poisson-inputs
# poisson rate for each input group when active (will be converted to Hz)
rate = 200
# number of input groups
N_inputs=10
# number of random switches between input groups
steps =  10
# duration of epochs between switches
step_dt =  200*ms
# array of poisson rates, will be transposed after shuffle
t_array=np.zeros((N_inputs, steps))
np.fill_diagonal(t_array, 1)
# randomize the order of activation
np.random.shuffle(t_array)
rate_array = rate*t_array.T
field_pair_i = 5
field_array = P.simulation['field_mags'][P.simulation['field_polarities'].index('anodal')]*t_array[field_pair_i,:].T
# timed arrays, first dimension should be time
# input_timed_array = TimedArray(rate_array*Hz, dt=step_dt)
# field_timed_array = TimedArray(field_array, dt=step_dt)

# set parameters for poisson inputs
#``````````````````````````````````
P.input['FF_train'] = {
    # number of inputs
    'N':N_inputs,
    # at each time step, array of rates of length N
    'poisson_rates':'input_timed_array(t,i)',
    # input timed array
    'input_timed_array':'',
    # 
    # FIXME RECORD SPIKES FROM EACH INPUT
    'rec_variables':False,
    'rec_indices':True
}
P.input['FF_test'] = {
    # number of inputs
    'N':N_inputs,
    # at each time step, array of rates of length N
    'poisson_rates':'input_timed_array(t,i)',
    # input timed array
    'input_timed_array':'',
    # 
    # FIXME RECORD SPIKES FROM EACH INPUT
    'rec_variables':False,
    'rec_indices':True
}
# delete unused parameters
del P.input['1']

# create input brian objects (SpikeGeneratorGroups)
input_paths={}
for path, params in P.input.iteritems():
    params['name']='inputs_'+path
    # FIXME SET NAMESPACE IN POISSON GROUP
    input_paths[path] = inputs._poisson(params)
#===========================================================================


neurons={}
synapses={}
# excitatory neurons
#============================================================================
neuron_group='E'
P.neurons[neuron_group]=copy.deepcopy(P.neurons['1'])
P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
P.neurons[neuron_group]['N']=N_E
nparams=P.neurons[neuron_group]
neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
    )
#===========================================================================

# inhibitory neurons
#===========================================================================
neuron_group='I'
P.neurons[neuron_group]=copy.deepcopy(P.neurons['1'])
P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
P.neurons[neuron_group]['N']=N_I
nparams=P.neurons[neuron_group]
neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
    )
#============================================================================

# del P.neurons['1']
# del P.init_neurons['1']

# parameter updates to be distributed to all synapses
#===========================================================================
P.synapses['1']['update_ampa_online']=0
P.synapses['1']['update_gaba_online']=0
P.synapses['1']['update_nmda_online']=0

# recurrent EE synapses
#===========================================================================
synapse_group='EE'
# copy default parameters
P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
# connect all to all
P.synapses[synapse_group]['connect_condition'] =  'True'
# synapse parameters
sparams = P.synapses[synapse_group]
# make synapses
synapses[synapse_group] = Synapses(neurons['E'], neurons['E'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
# connect synapses
synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])

# initialize uniform weights 
#```````````````````````````
# number of synapses
Nsyn = len(synapses[synapse_group].i)
# initial weight matrices
P.init_synapses[synapse_group]['w_ampa'] = param._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)
P.init_synapses[synapse_group]['w_clopath'] = param._weight_array_uniform(Nsyn=Nsyn, w=w_EE_init)
#==========================================================================


# recurrent EI synapses
#===========================================================================
synapse_group='EI'
# copy default parameters
P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
P.synapses[synapse_group]['update_ampa_online']=0
P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
# connect all to all
P.synapses[synapse_group]['connect_condition'] =  'rand()<.8' # FIXME each I neuron should receive input from 8 E neurons
# synapse parameters
sparams = P.synapses[synapse_group]
# make synapses
synapses[synapse_group] = Synapses(neurons['E'], neurons['I'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
# connect synapses
synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])
# initialize uniform weights 
#```````````````````````````
# number of synapses
Nsyn = len(synapses[synapse_group].i)
# initial weight matrices
P.init_synapses[synapse_group]['w_ampa'] = param._weight_array_uniform(Nsyn=Nsyn, w=1)
P.init_synapses[synapse_group]['w_clopath'] = param._weight_array_uniform(Nsyn=Nsyn, w=1)
#==========================================================================


# recurrent IE synapses
#===========================================================================
# FIXME how to iniitialize inhibitory synapses and toggle online plasticity on and off
synapse_group='IE'
# copy default parameters
P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
P.synapses[synapse_group]['update_ampa_online']=0
P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
# FIXME REMOVE PRESYNAPTIC ADAPTATION FROM INHIBITORY INPUTS?
# connect all to all
P.synapses[synapse_group]['connect_condition'] =  'rand()<.6' # FIXME each I neuron should project to 6 random E neurons
# synapse parameters
sparams = P.synapses[synapse_group]
# make synapses
synapses[synapse_group] = Synapses(neurons['I'], neurons['E'], Eq.synapse_i, on_pre=Eq.synapse_i_pre, on_post=Eq.synapse_i_post, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
# connect synapses
synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])
# initialize uniform weights 
#```````````````````````````
# number of synapses
Nsyn = len(synapses[synapse_group].i)
# initial weight matrices
P.init_synapses[synapse_group]['w_gaba'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
P.init_synapses[synapse_group]['w_vogels'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
#==========================================================================


# feedforward synapses on to E neurons during training
#==========================================================================
# recurrent EE synapses
synapse_group='FE_train'
# copy default parameters
P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
# connect all to all
P.synapses[synapse_group]['connect_condition'] = 'abs(i-j)<='+str(N_assembly)
# synapse parameters
sparams = P.synapses[synapse_group]
# create synapses
# print input_paths['FF']
# print neurons['E']
synapses[synapse_group] = Synapses(input_paths['FF_train'], neurons['E'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])
# initialize uniform weights 
#```````````````````````````
# number of synapses
Nsyn = len(synapses[synapse_group].i)
# initial weight matrices
P.init_synapses[synapse_group]['w_ampa'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
P.init_synapses[synapse_group]['w_clopath'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
#===========================================================================


# feedforward synapses on to E neurons during testing
#==========================================================================
# recurrent EE synapses
synapse_group='FE_test'
# copy default parameters
P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
# connect all to all
P.synapses[synapse_group]['connect_condition'] = 'abs(i-j)<='+str(N_recall)
# synapse parameters
sparams = P.synapses[synapse_group]
# create synapses
synapses[synapse_group] = Synapses(input_paths['FF_test'], neurons['E'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
# connect synapses
synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])
# initialize uniform weights 
#```````````````````````````
# number of synapses
Nsyn = len(synapses[synapse_group].i)
# initial weight matrices
P.init_synapses[synapse_group]['w_ampa'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
P.init_synapses[synapse_group]['w_clopath'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
#===========================================================================


# feedforward synapses on to I neurons
#==========================================================================
# recurrent EE synapses
synapse_group='FI_train'
# copy default parameters
P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
P.synapses[synapse_group]['update_ampa_online']=0
P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
# no connect condition, specify i and j directly
P.synapses[synapse_group]['connect_condition'] = None 
# for each feedforward input neuron, randomly select an inhibitory neuron in the network
P.synapses[synapse_group]['connect_j'] = np.random.randint(low=0, high=P.neurons[neuron_group]['N']-1, size=P.input['FF_train']['N'])
P.synapses[synapse_group]['connect_i'] = np.arange(P.input['FF_train']['N'])
# synapse parameters
sparams = P.synapses[synapse_group]
# create synapses
synapses[synapse_group] = Synapses(input_paths['FF_train'], neurons['I'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
# connect synapses
synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], i=P.synapses[synapse_group]['connect_i'], j=P.synapses[synapse_group]['connect_j'])
# initialize uniform weights 
#```````````````````````````
# number of synapses
Nsyn = len(synapses[synapse_group].i)
# initial weight matrices
P.init_synapses[synapse_group]['w_ampa'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
P.init_synapses[synapse_group]['w_clopath'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
#=========================================================================

# feedforward synapses on to I neurons
#==========================================================================
# recurrent EE synapses
synapse_group='FI_test'
# copy default parameters
P.synapses[synapse_group]=copy.deepcopy(P.synapses['1'])
P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
# no connect condition, specify i and j directly
P.synapses[synapse_group]['connect_condition'] = None 
# for each feedforward input neuron, randomly select an inhibitory neuron in the network
P.synapses[synapse_group]['connect_j'] = np.random.randint(low=0, high=P.neurons[neuron_group]['N']-1, size=P.input['FF_test']['N'])
P.synapses[synapse_group]['connect_i'] = np.arange(P.input['FF_test']['N'])
# synapse parameters
sparams = P.synapses[synapse_group]
# create synapses
synapses[synapse_group] = Synapses(input_paths['FF_test'], neurons['I'], Eq.synapse_e, on_pre=Eq.synapse_e_pre_nonadapt, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
# connect synapses
synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'], i=P.synapses[synapse_group]['connect_i'], j=P.synapses[synapse_group]['connect_j'])
# initialize uniform weights 
#```````````````````````````
# number of synapses
Nsyn = len(synapses[synapse_group].i)
# initial weight matrices
P.init_synapses[synapse_group]['w_ampa'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
P.init_synapses[synapse_group]['w_clopath'] = param._weight_array_uniform(Nsyn=Nsyn, w=1.)
#=========================================================================

# initial conditions
#===================================================================
# P.init_synapses['2']=copy.deepcopy(P.init_synapses['1'])
run_control._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
run_control._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

# set up recording
#====================================================================
# recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
P.neurons['E']['rec_variables'].append('I_field')
rec = run_control._build_state_rec(brian_objects=[neurons, synapses,], keys=['neurons', 'synapses',], P=P)

# set up network
#======================================================================
net = Network()
net = run_control._collect_brian_objects(net, input_paths, neurons, synapses, rec['neurons'], rec['synapses'])

# run simulation
#=======================================================================
# set time step
defaultclock.dt = P.simulation['dt']
# store initialized network state
net.store('initial')

# dictionary for group data over multiple trials
train_group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename_train, df=True)
test_group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename_test, df=True)

# FIXME 
# FOR REPEATED SIMULATIONS ARE POISSON INPUTS REGENEREATED?
# FREEZE WEIGHTS AND REACTIVATE NEURONS AFTER TRAIN

P.simulation['run_time'] = steps*step_dt
# set number of trials
P.simulation['trials']=1
for trial in range(P.simulation['trials']):
    # restore initial conditions after each trial
    net.restore('initial')

    # Training
    #====================================================================
    # set ampa weights to be plastic
    # synapses['EE'].update_ampa_online =1
    P.synapses['EE']['update_ampa_online']=1
    P.synapses['FE_train']['update_ampa_online']=1
    P.synapses['FE_test']['update_ampa_online']=0
    
    # create two timed arrays, one for training and one for test
    # when training, the test array is all zeros and vice versa
    # reshuffle timed arrray values
    active_arrays={}
    inactive_arrays={}
    np.random.shuffle(t_array)
    rate_array = rate*t_array.T
    field_pair_i = 5
    field_array = P.simulation['field_mags'][P.simulation['field_polarities'].index('anodal')]*t_array[field_pair_i,:].T
    active_arrays['input_timed_array'] = TimedArray(rate_array*Hz, dt=step_dt)
    active_arrays['field_timed_array'] = TimedArray(field_array, dt=step_dt)
    inactive_arrays['input_timed_array'] = TimedArray(np.zeros(rate_array.shape)*Hz, dt=step_dt)
    inactive_arrays['field_timed_array'] = TimedArray(np.zeros(field_array.shape)*Hz, dt=step_dt)

    # print active_arrays['field_timed_array'].values
    # make sure timed arrays are available to the appropriate namespace
    P.input['FF_train']['input_timed_array'] = active_arrays['input_timed_array']
    P.input['FF_test']['input_timed_array'] = inactive_arrays['input_timed_array']
    P.neurons['E']['field_timed_array'] = active_arrays['field_timed_array']
    
    # store randomized initial condition
    net.store('randomized')
    
    # generate unique trial id
    P.simulation['trial_id'] = str(uuid.uuid4())

    # set electric field in parameter dictionaries
    P.simulation['field_mag'] = P.simulation['field_mags'][P.simulation['field_polarities'].index('anodal')]

    P.simulation['field_polarity'] = 'anodal'
    P.simulation['field_color'] = P.simulation['field_colors'][P.simulation['field_polarities'].index('anodal')]

    # FIX
    # FIXME check if neuron objects have access to top namespace 
    # only add field to excitatory neurons 
    P.neurons['E']['I_field'] = 'field_timed_array(t)'
    neurons['E'].I_field = P.neurons['E']['I_field']

    net.run(P.simulation['run_time'])

    print 'first run finished'

    # get trained weights
    trained_weights = {}
    weight_keys = ['w_ampa','w_nmda', 'w_gaba', 'w_clopath','w_vogels']
    for syn_group, syn in synapses.iteritems():
        trained_weights[syn_group]={}
        for weight_key in weight_keys:
            if hasattr(syn, weight_key):
                trained_weights[syn_group][weight_key] = getattr(syn, weight_key)[-1]

    # training data
    train_df = analysis._rec2df(rec=rec, P=P, include_P=False)

    # Test
    #==================================================================
    # restore randomized network
    net.restore('randomized')

    # set ampa weights to be fixed
    # synapses['EE'].update_ampa_online = 0
    P.synapses['EE']['update_ampa_online']=0
    P.synapses['FE_train']['update_ampa_online']=0
    P.synapses['FE_test']['update_ampa_online']=0

    print synapses['EE'].namespace
    # initialize weights to trained values
    for syn_group, syn in synapses.iteritems():
        if hasattr(syn, 'w_ampa'):
            synapses[syn_group].w_ampa=trained_weights[syn_group]['w_clopath']
            synapses[syn_group].w_clopath=trained_weights[syn_group]['w_clopath']
        if hasattr(syn, 'w_gaba'):
            synapses[syn_group].w_gaba=trained_weights[syn_group]['w_vogels']
            synapses[syn_group].w_vogels=trained_weights[syn_group]['w_vogels']

    # FIXME ACTIVATE SEPARATE POISSON INPUTS FOR TEST PHASE
    # make sure timed arrays are available to the appropriate namespace
    P.input['FF_train']['input_timed_array'] = inactive_arrays['input_timed_array']
    P.input['FF_test']['input_timed_array'] = active_arrays['input_timed_array']
    P.neurons['E']['field_timed_array'] = inactive_arrays['field_timed_array']

    # run simulation
    net.run(P.simulation['run_time'])

    # convert recorded data to pandas dataframe
    test_df = analysis._rec2df(rec=rec, P=P, include_P=False)

    # add to group data
    train_group_df = train_group_df.append(train_df, ignore_index=True)
    test_group_df = test_group_df.append(test_df, ignore_index=True)


# save data
train_group_df.to_pickle(group_data_directory+group_data_filename_train)
test_group_df.to_pickle(group_data_directory+group_data_filename_test)