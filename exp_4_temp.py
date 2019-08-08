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
group_data_directory = 'Data/'+__name__+'/'
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

# free parameters
# rates and weights of feedforward poisson inputs
# timescale and set point of homeostatic plasticity
# weights of IE synapses


# load equations for adaptive exponential integrate and fire neuron
#=====================================================================
Eq = equations.AdexBonoClopath()
#=====================================================================

# design feedforward input
#=======================================================================
# set up feedforward inputs 
# see :https://brian2.readthedocs.io/en/stable/user/input.html#setting-rates-for-poisson-inputs
# poisson rate for each input group when active (will be converted to Hz)
rate = 100
# number of random switches between input groups
steps =  10
# duration of epochs between switches
step_dt =  100*ms
# number of input groups
N_inputs=10
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
    'N':10,
    # at each time step, array of rates of length N
    'poisson_rates':1*Hz,#'input_timed_array(:,t)',
    # input timed array
    'input_timed_array':'',
    # 
    # FIXME RECORD SPIKES FROM EACH INPUT
    'rec_variables':True,
    'rec_indices':True
}
P.input['FF_test'] = {
    # number of inputs
    'N':10,
    # at each time step, array of rates of length N
    'poisson_rates':1*Hz,#'input_timed_array(:,t)',
    # input timed array
    'input_timed_array':'',
    # 
    # FIXME RECORD SPIKES FROM EACH INPUT
    'rec_variables':True,
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
P.neurons[neuron_group]['N']=10
nparams=P.neurons[neuron_group]
neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
    )
#===========================================================================

# inhibitory neurons
#===========================================================================
neuron_group='I'
P.neurons[neuron_group]=copy.deepcopy(P.neurons['1'])
P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
P.neurons[neuron_group]['N']=3
nparams=P.neurons[neuron_group]
neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
    )

# initial conditions
#===================================================================
# P.init_synapses['2']=copy.deepcopy(P.init_synapses['1'])
run_control._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
run_control._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

# set up recording
#====================================================================
# recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
rec = run_control._build_state_rec(brian_objects=[neurons, synapses], keys=['neurons', 'synapses'], P=P)

# set up network
#======================================================================
net = Network()
net = run_control._collect_brian_objects(net, input_paths, neurons, synapses, rec['neurons'], rec['synapses'])
net = run_control._collect_brian_objects(net, input_paths, neurons, rec['neurons'],)

# run simulation
#=======================================================================
# set time step
defaultclock.dt = P.simulation['dt']
# store initialized network state
net.store('initial')

# dictionary for group data over multiple trials
group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename, df=True)

# FIXME 
# FOR REPEATED SIMULATIONS ARE POISSON INPUTS REGENEREATED?
# FREEZE WEIGHTS AND REACTIVATE NEURONS AFTER TRAIN

P.simulation['run_time'] = steps*step_dt
# set number of trials
P.simulation['trials']=1
for trial in range(P.simulation['trials']):
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

    # make sure timed arrays are available to the appropriate namespace
    P.input['FF_train']['input_timed_array'] = active_arrays['input_timed_array']
    P.input['FF_test']['input_timed_array'] = inactive_arrays['input_timed_array']
    P.neurons['E']['field_timed_array'] = active_arrays['field_timed_array']
    net.run(P.simulation['run_time'])