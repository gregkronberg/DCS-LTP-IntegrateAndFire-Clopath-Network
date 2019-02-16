'''
simulate network of 10 E and 3 I neurons as in clopath 2010
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

# directory and file name to store data
group_data_directory = 'Data/'+__name__+'/'
group_data_filename = __name__+'_data.pkl'


# load parameters
P = param.Default()
Pclopath=  param.Clopath2010()


P.synapses['1'] = Pclopath.synapses['1']
P.neurons['1'] = Pclopath.neurons['1']

# P.synapses['1']['A_LTP'] = 1000*P.synapses['1']['A_LTP']
# P.synapses['1']['A_LTD'] = 1000*P.synapses['1']['A_LTD']
# P.synapses['1']['include']
# convert parameters to dictionary
P_dict = P.__dict__
# load equations for adaptive exponential integrate and fire neuron
Eq = equations.AdexBonoClopath()

# input parameters
#=======================================================================
# strong
# set up feedforward inputs see :https://brian2.readthedocs.io/en/stable/user/input.html#setting-rates-for-poisson-inputs
rate = 100
steps =  10
step_dt =  100*ms
N_inputs=10
rate_array=np.zeros((N_inputs, steps))
np.fill_diagonal(rate_array, rate)
# randomize the order of activation
np.random.shuffle(rate_array)
input_timed_array = TimedArray(rate_array*Hz, dt=step_dt)
# design poi
P.input['FF']=P.input['1']
del P.input['1']
P.input['FF'] = {
    # input/stimulation parameters
    #============================================================================
    'n_poisson':10,
    'poisson_rates':'input_timed_array(:,t)'

    'rec_variables':[],
    'rec_indices':True
}

# create input brian objects (SpikeGeneratorGroups)
input_paths={}
for path, params in P.input.iteritems():
    params['name']='inputs_'+path
    input_paths[path] = inputs._poisson(params)

# excitatory neurons
neuron_group='E'
P.neurons[neuron_group]=copy.deepcopy(P.neurons['1'])
P.neurons[neuron_group]['N']=10
nparams=P.neurons[neuron_group]
neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
    )

# inhibitory neurons
neuron_group='I'
P.neurons[neuron_group]=copy.deepcopy(P.neurons['1'])
P.neurons[neuron_group]['N']=3
nparams=P.neurons[neuron_group]
neurons[neuron_group] =  NeuronGroup(nparams['N'], Eq.neuron, threshold=nparams['threshold_condition'], reset=Eq.neuron_reset,   refractory=nparams['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=nparams
    )

del P.neurons['1']


# feedforward synapses

# post synaptic neuron
#=======================================================================
# create 3 identical neuron groups: neuron 1=(strong, none), neuron 2=(weak, none), neuron 3=(strong, weak)

# dictionaries for storing neuron and synapse objects
neurons={}
synapses={}
# for each combination of inputs, create a neuron group
for combo_i, combo in enumerate(list(input_combos)):
    # neuron group name
    neuron_group = str(combo_i+1)
    # copy parameters to each neuron
    P.neurons[neuron_group] = copy.deepcopy(P.neurons['1'])
    P.neurons[neuron_group]['N']=1
    params=P.neurons[neuron_group]
    print params['include_homeostatic']
    neurons[neuron_group] =  NeuronGroup(P.neurons[neuron_group]['N'], Eq.neuron, threshold=params['threshold_condition'], reset=Eq.neuron_reset,   refractory=params['refractory_time'],  method='euler', name='neurons_'+neuron_group, namespace=params
    )
    # print neurons[neuron_group].include_homeostatic
    P.init_neurons[neuron_group]=copy.deepcopy(P.init_neurons['1'])
    for input_path in combo:
        synapse_group = 'pre_'+input_path+'_post_'+neuron_group
        P.synapses[synapse_group] = copy.deepcopy(P.synapses['1'])
        P.synapses[synapse_group]['connect_condition']='True'
        P.init_synapses[synapse_group]=copy.deepcopy(P.init_synapses['1'])
        synapses[synapse_group] = Synapses(input_paths[input_path], neurons[neuron_group], Eq.synapse_e, on_pre=Eq.synapse_e_pre, namespace=P.synapses[synapse_group], name='synapses_'+synapse_group, method=euler)
        synapses[synapse_group].connect(condition=P.synapses[synapse_group]['connect_condition'])

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

# run simulation
#=======================================================================
# set time step
defaultclock.dt = P.simulation['dt']
# store initialized network state
net.store('initial')

# set number of pre and post synaptic neurons
Npre_1=1
Npost_1=1

# dictionary for group data over multiple trials
group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename, df=True)

# set number of trials
P.simulation['trials']=10
for trial in range(P.simulation['trials']):
    # restore initial conditions after each trial
    net.restore('initial')
    
    for synapse_group in synapses:

        # FIXME dynamically get number of pre and postsynaptic neurons for each group
        P.init_synapses[synapse_group]['w_ampa'] = param._weight_matrix_randn(Npre=Npre_1, Npost=Npost_1, w_mean=.9, w_std=0.2)

        synapses[synapse_group].w_ampa = P.init_synapses[synapse_group]['w_ampa']
    
    # store randomized initial condition
    net.store('randomized')
    
    # generate unique trial id
    P.simulation['trial_id'] = str(uuid.uuid4())

    # iterate over electric field conditions
    for field_i, field in enumerate(P.simulation['field_mags']):

        # reset network
        net.restore('randomized')

        # set electric field in parameter dictionaries
        P.simulation['field_mag'] = field
        P.simulation['field_polarity'] = P.simulation['field_polarities'][field_i]
        P.simulation['field_color'] = P.simulation['field_colors'][field_i]
        for neuron_group in neurons:
            P.neurons[neuron_group]['I_field'] = field

            # set electric field in brian
            neurons[neuron_group].I_field= field

        # run simulation
        net.run(P.simulation['run_time'])

        # convert recorded data to pandas dataframe
        data_df = analysis._rec2df(rec=rec, P=P)

        # add to group data
        group_df = group_df.append(data_df, ignore_index=True)


# save data
group_df.to_pickle(group_data_directory+group_data_filename)