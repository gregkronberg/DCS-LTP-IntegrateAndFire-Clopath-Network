'''
simulate two pathway experiments with DCS and theta burst LTP
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

# load parameters
P = param.Default()
# convert to dictionary
P_dict = P.__dict__
# load equations for adaptive exponential integrate and fire neuron
Eq = equations.AdexBonoClopath()

# neurons
#=====================================
neurons={}
P.neurons['1']['N']=1
neurons['1']= NeuronGroup(P.neurons['1']['N'], Eq.neuron , threshold=P.neurons['1']['threshold_condition'], reset=Eq.neuron_reset,   refractory=P.neurons['1']['refractory_time'],  method='euler', namespace=P.neurons['1']
    )

# input synapses
#=======================================
input_path={}
input_path['1'] = inputs._tbs(P.input['1'])
P.synapses['1']['connect_condition']='True'

synapses = {}
synapses['1'] = Synapses(input_path['1'], neurons['1'], Eq.synapse_e, on_pre=Eq.synapse_e_pre, namespace=P.synapses['1'])
synapses['1'].connect(condition=P.synapses['1']['connect_condition'])
Npre_1 = len(list(set(synapses['1'].i)))
Npost_1= len(list(set(synapses['1'].j)))
P.init_synapses['1']['w_ampa'] = param._weight_matrix_uniform(Npre=Npre_1, Npost=Npost_1, w=1)

# recurrent synapses
#=========================================
P.synapses['2'] = copy.deepcopy(P.synapses['1'])
P.synapses['2']['connect_condition'] = 'j==0'

P.init_synapses['2'] = copy.deepcopy(P.init_synapses['1'])


synapses['2'] = Synapses(neurons['1'],neurons['1'], Eq.synapse_e, on_pre=Eq.synapse_e_pre, namespace=P.synapses['2'])
synapses['2'].connect(condition=P.synapses['2']['connect_condition'])
Npre_2 = len(list(set(synapses['2'].i)))
Npost_2= len(list(set(synapses['2'].j))) 
P.synapses['2']['w_matrix'] = param._weight_matrix_randn(Npre=Npre_2, Npost=Npost_2, w_mean=0.5, w_std=0.2)
P.synapses['2']['w_vector'] = param._broadcast_weight_matrix(w_matrix=P.synapses['2']['w_matrix'], i=synapses['2'].i, j=synapses['2'].j)
P.init_synapses['2']['w_ampa'] = P.synapses['2']['w_vector']

# # initial conditions
# #===================================================================
run_control._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
run_control._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

# set up recording
#====================================================================
rec= {}
rec['neurons']={}
rec['synapses']={}
# rec['input_path']={}

for obj_type, obj in rec.iteritems():
    for group_key, group in globals()[obj_type].iteritems():
        brian_object = globals()[obj_type][group_key]
        rec[obj_type][group_key] = StateMonitor(brian_object, P_dict[obj_type][group_key]['rec_variables'], record=True)

# set up network
#======================================================================
net = Network()
net = run_control._collect_brian_objects(net, input_path, neurons, synapses, rec['neurons'], rec['synapses'])

# run simulation
#=======================================================================
defaultclock.dt = P.simulation['dt']
net.store('initial')


data_mon = []
data_mon_p = []
data_frame = pd.DataFrame()

group_data = {}
# run(run_time)
P.simulation['trials']=2
for trial in range(P.simulation['trials']):
    P.init_synapses['1']['w_ampa'] = param._weight_matrix_randn(Npre=Npre_1, Npost=Npost_1, w_mean=1, w_std=0.5)
    synapses['1'].w_ampa = P.init_synapses['1']['w_ampa']
    P.simulation['trial_id'] = str(uuid.uuid4())
    for field_i, field in enumerate(P.simulation['field_mags']):
        P.simulation['field_mag'] = field
        P.neurons['1']['I_field'] = field
        net.restore('initial')
        neurons['1'].I_field= field
        net.run(P.simulation['run_time'])
        data_dict = analysis._rec2dict(rec=rec, P=P)
        group_data = analysis._add_to_group_data(group_data=group_data, data_dict=data_dict)

group_frames = analysis._dict2frame(data_dict=group_data)
