from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy

def _rec2dict(rec, P):
    '''
    '''
    # FIXME store data as pandas series instead of numpy arrays to facilitate conversion to dataframe

    # convert 2d array to series of row vectors: pd.Series(2darray.tolist())

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

    group_data={}
    # iterate over type of recorded object
    for group_type_key, group_type in rec.iteritems():
        if group_type_key not in group_data:
            group_data[group_type_key] = {}
        # iterate over groups
        for group_key, group in group_type.iteritems():

            for var in group.record_variables:
                
                data_array = getattr(group, var)

                n = data_array.shape[0]

                if group_type_key == 'synapses':
                    pre_index = group.source.i
                    post_index = group.source.j
                else:
                    pre_index = np.array([np.nan]*n)
                    post_index = np.array([np.nan]*n)

                index = group.record
                brian_group_name = np.array([group.source.name]*n)
                group_name = np.array([group_key]*n)
                trial_id = np.array([P.simulation['trial_id']]*n)
                field_mag = np.array([P.simulation['field_mag']]*n)
                P_array = np.array([P]*n)
                
                group_data[group_type_key][var] = {
                'data':data_array,
                'index':index,
                'pre_index':pre_index,
                'post_index':post_index,
                'brian_group_name':brian_group_name,
                'group_name':group_name,
                'trial_id':trial_id,
                'P':P_array,
                'field_mag':field_mag,

                }

    return group_data

def _add_to_group_data(group_data, data_dict):
    '''
    '''
    for group_type_key, group_type in data_dict.iteritems():
        if group_type_key not in group_data:
            group_data[group_type_key] = data_dict[group_type_key]
            continue
        for variable_key, variable in group_type.iteritems():
            if variable_key not in group_data[group_type_key]:
                group_data[group_type_key][variable_key] = variable
                continue
            for data_key, data in variable.iteritems():
                if data_key not in group_data[group_type_key][variable_key]:
                    group_data[group_type_key][variable_key][data_key] = data
                else:
                    print data_key, group_data[group_type_key][variable_key][data_key]
                    print data_key, data
                    group_data[group_type_key][variable_key][data_key] = np.append(group_data[group_type_key][variable_key][data_key], data, axis=0)

    return group_data
