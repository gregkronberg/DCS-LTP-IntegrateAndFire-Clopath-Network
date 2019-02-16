'''
qualitatively reproduce two pathway effects of DCS on theta burst LTP in single compartment integrate and fire neuron
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

# directory and file name to store data
group_data_directory = 'Data/'+__name__+'/'
group_data_filename = __name__+'_data.pkl'

# load parameters
P = param.Default()
# convert parameters to dictionary
P_dict = P.__dict__
# load equations for adaptive exponential integrate and fire neuron
Eq = equations.AdexBonoClopath()

# post synaptic neuron
#=======================================================================
neurons={}
P.neurons['1']['N']=1
neurons['1']= NeuronGroup(P.neurons['1']['N'], Eq.neuron , threshold=P.neurons['1']['threshold_condition'], reset=Eq.neuron_reset,   refractory=P.neurons['1']['refractory_time'],  method='euler', namespace=P.neurons['1']
    )

# input synapses
#=======================================================================
P.input['2'] = {
    # input/stimulation parameters
    #============================================================================
    'pulses' : 1,
    'bursts' : 4,
    'pulse_freq' : 100,
    'burst_freq' : 5,
    'warmup' : 20,

    'I_input':0*pA,

    'rec_variables':[],
    'rec_indices':True
}

input_path={}
# theta burst input
input_path['1'] = inputs._tbs(P.input['1'])
# weak 5 Hz input
input_path['2'] = inputs._tbs(P.input['2'])
# connect all inputs to all postsynaptic neurons
P.synapses['1']['connect_condition']='True'
P.synapses['2']=copy.deepcopy(P.synapses['1'])
# build and connect synapses
synapses = {}
synapses['1'] = Synapses(input_path['1'], neurons['1'], Eq.synapse_e, on_pre=Eq.synapse_e_pre, namespace=P.synapses['1'])
synapses['1'].connect(condition=P.synapses['1']['connect_condition'])
synapses['2'] = Synapses(input_path['2'], neurons['1'], Eq.synapse_e, on_pre=Eq.synapse_e_pre, namespace=P.synapses['2'])
synapses['2'].connect(condition=P.synapses['2']['connect_condition'])

# initial conditions
#===================================================================
P.init_synapses['2']=copy.deepcopy(P.init_synapses['1'])
run_control._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
run_control._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

# set up recording
#====================================================================
# recording dictionary as rec{object type}{group key}[state monitor], e.g. rec['neurons']['1'][StateMonitor]
rec = run_control._build_state_rec(brian_objects=[neurons, synapses], keys=['neurons', 'synapses'], P=P)


# rec= {}
# rec['neurons']={}
# rec['synapses']={}

# # iterate over object types (neurons or synapses)
# for obj_type, obj in rec.iteritems():
#   # iterate over group
#   for group_key, group in globals()[obj_type].iteritems():
#       # get underlying brian object
#       brian_object = globals()[obj_type][group_key]
#       # setup state monitor
#       rec[obj_type][group_key] = StateMonitor(brian_object, P.__dict__[obj_type][group_key]['rec_variables'], record=True)

# set up network
#======================================================================
net = Network()
net = run_control._collect_brian_objects(net, input_path, neurons, synapses, rec['neurons'], rec['synapses'])

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
P.simulation['trials']=3
for trial in range(P.simulation['trials']):
    # restore initial conditions after each trial
    net.restore('initial')
    
    # randomize input weights
    P.init_synapses['1']['w_ampa'] = param._weight_matrix_randn(Npre=Npre_1, Npost=Npost_1, w_mean=1, w_std=0.5)
    P.init_synapses['2']['w_ampa'] = param._weight_matrix_randn(Npre=Npre_1, Npost=Npost_1, w_mean=1, w_std=0.5)
    
    # set initial weights
    synapses['1'].w_ampa = P.init_synapses['1']['w_ampa']
    synapses['2'].w_ampa = P.init_synapses['2']['w_ampa']
    
    # store randomized initial condition
    net.store('randomized')
    
    # generate unique trial id
    P.simulation['trial_id'] = str(uuid.uuid4())

    # iterate over electric field conditions
    for field_i, field in enumerate(P.simulation['field_mags']):

        # set electric field in parameter dictionaries
        P.simulation['field_mag'] = field
        P.neurons['1']['I_field'] = field

        # reset network
        net.restore('randomized')

        # set electric field in brian
        neurons['1'].I_field= field

        # run simulation
        net.run(P.simulation['run_time'])

        # convert recorded data to pandas dataframe
        data_df = analysis._rec2df(rec=rec, P=P)

        # add to group data
        group_df = group_df.append(data_df, ignore_index=True)


# save data
group_df.to_pickle(group_data_directory+group_data_filename)