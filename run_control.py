from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy

def _set_initial_conditions(brian_object, init_dic):
    '''
    '''
    if isinstance(brian_object, dict):
        for group_key, group in brian_object.iteritems():
            for param, val in init_dic[group_key].iteritems():
                setattr(brian_object[group_key], param, val)

    else:
        for param, val in init_dic[group_key].iteritems():
            setattr(brian_object, param, val)

def _collect_brian_objects(net, *dics):
    '''
    '''
    for object_container in dics:
        for group_key, group in object_container.iteritems():
            net.add(object_container[group_key])

    return net


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