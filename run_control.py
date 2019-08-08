from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy

def _build_spike_rec(brian_objects, keys, P):
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

def _build_state_rec(brian_objects, keys, P):
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

def _set_initial_conditions(brian_object, init_dic):
    '''
    '''
    if isinstance(brian_object, dict):
        for group_key, group in brian_object.iteritems():
            for param, val in init_dic[group_key].iteritems():
                if hasattr(brian_object[group_key], param):
                    print group_key, getattr(brian_object[group_key], param).shape
                    setattr(brian_object[group_key], param, val)

    else:
        for param, val in init_dic[group_key].iteritems():
            if hasattr(brian_object, param):
                setattr(brian_object, param, val)

def _collect_brian_objects(net, *dics):
    '''
    '''
    for object_container in dics:
        for group_key, group in object_container.iteritems():
            net.add(object_container[group_key])

    return net

def _rec2dict(rec, P):
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

def _monitor_to_dataframe(mon, P):
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







class Setup:
    '''
    '''
    def __init__(self, P):
        '''
        '''
        self.P=P

    def standard(self, P, **kwargs):
        '''
        '''

        # create neuron group
        #====================================================================
        # nrn = NeuronGroup(N, eq_nrn , threshold=threshold_condition, reset=eq_nrn_reset, refractory=refractory_time, method='euler')
        Eq = Eqs()
        self.nrn = NeuronGroup(N, Eq.nrn , threshold=threshold_condition, reset=Eq.nrn_reset,   refractory=refractory_time,  method='euler')


        # inputs to network
        #====================================================================
        self.input_nrn = Inputs()._tbs(P.p)

        # create synapses
        #====================================================================
        self.eq_syn =  Eq.syn_stp + '\n' + Eq.syn_clopath  

        self.eq_syn_pre = Eq.syn_ampa_pre + '\n' + Eq.syn_nmda_pre + '\n'+ Eq.syn_stp_pre + '\n' + Eq.syn_clopath_pre

        self.input_syn = Synapses(self.input_nrn, self.nrn, self.eq_syn, on_pre=self.eq_syn_pre,)
        self.input_syn.connect(condition=P.p['input_syn_condition'])

        # setup recording variables
        #===================================================================

        self.rec={}
        self.rec['nrn'] = StateMonitor(self.nrn, P.p['rec_variables_nrn'], record=True)
        self.rec['nrn_spikes'] = SpikeMonitor(self.nrn)
        self.rec['syn'] = StateMonitor(self.input_syn, P.p['rec_variables_input_syn'], record=True)
        self.rec['input'] = SpikeMonitor(self.input_nrn, record=True)

        # initial conditions
        #===================================================================
        for param, val in P.init_nrn.iteritems():
            setattr(self.nrn, param, val)

        for param, val in P.init_input_syn.iteritems():
            setattr(self.input_syn, param, val)

        self.net = Network(self.nrn, self.input_nrn, self.input_syn, self.rec)

        return self.net

class Run:
    def __init__(self, ):
        '''
        '''

    def _run(self, net, P,):
        '''
        '''
        # run
        #=======================================================================
        run_time = P.p['run_time']
        net.store()

        # run(run_time)
        for trial in range(trials):
            net.restore()
            net.run(run_time, namespace=P.p)