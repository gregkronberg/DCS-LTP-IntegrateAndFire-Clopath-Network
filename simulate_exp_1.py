'''
qualitatively reproduce single pathway effects of DCS on theta burst LTP in single compartment integrate and fire neuron
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

# post synaptic neuron
#=====================================
neurons={}
P.neurons['1']['N']=1
neurons['1']= NeuronGroup(P.neurons['1']['N'], Eq.neuron , threshold=P.neurons['1']['threshold_condition'], reset=Eq.neuron_reset,   refractory=P.neurons['1']['refractory_time'],  method='euler', namespace=P.neurons['1']
	)

# input synapses
#=======================================
input_path={}
# theta burst input
input_path['1'] = inputs._tbs(P.input['1'])
# connect all inputs to all postsynaptic neurons
P.synapses['1']['connect_condition']='True'

synapses = {}
synapses['1'] = Synapses(input_path['1'], neurons['1'], Eq.synapse_e, on_pre=Eq.synapse_e_pre, namespace=P.synapses['1'])
synapses['1'].connect(condition=P.synapses['1']['connect_condition'])

# # initial conditions
# #===================================================================
run_control._set_initial_conditions(brian_object=neurons, init_dic=P.init_neurons)
run_control._set_initial_conditions(brian_object=synapses, init_dic=P.init_synapses)

# set up recording
#====================================================================
rec= {}
rec['neurons']={}
rec['synapses']={}

# iterate over object types (neurons or synapses)
for obj_type, obj in rec.iteritems():
	# iterate over group
	for group_key, group in globals()[obj_type].iteritems():
		# get underlying brian object
		brian_object = globals()[obj_type][group_key]
		# setup state monitor
		rec[obj_type][group_key] = StateMonitor(brian_object, P_dict[obj_type][group_key]['rec_variables'], record=True)

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

Npre_1=1
Npost_1=1
group_dict = {}
# run(run_time)
P.simulation['trials']=10
for trial in range(P.simulation['trials']):
	net.restore('initial')
	P.init_synapses['1']['w_ampa'] = param._weight_matrix_randn(Npre=Npre_1, Npost=Npost_1, w_mean=1, w_std=0.5)
	synapses['1'].w_ampa = P.init_synapses['1']['w_ampa']
	print synapses['1'].w_ampa
	net.store('randomized')
	synapses['1'].w_ampa = P.init_synapses['1']['w_ampa']
	P.simulation['trial_id'] = str(uuid.uuid4())
	for field_i, field in enumerate(P.simulation['field_mags']):
		P.simulation['field_mag'] = field
		P.neurons['1']['I_field'] = field
		net.restore('randomized')
		neurons['1'].I_field= field
		net.run(P.simulation['run_time'])
		data_dict = analysis._rec2dict(rec=rec, P=P)
		group_dict = analysis._add_to_group_data(group_data=group_dict, data_dict=data_dict)

group_df = analysis._dict2frame(data_dict=group_dict)



# for group_key, group in neurons.iter
# brian_objects = collect(level=0)

# net = Network(nrn, self.input_nrn, self.input_syn, self.rec)