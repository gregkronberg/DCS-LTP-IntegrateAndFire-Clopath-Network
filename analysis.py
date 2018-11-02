from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy

def _rec2dict(rec, P):
    ''' convert recording objects from brian 
    '''
    group_data={}
    # iterate over type of recorded object
    for group_type_key, group_type in rec.iteritems():
        # if group_type_key not in group_data:
        #     group_data[group_type_key] = {}
        # iterate over groups
        for group_key, group in group_type.iteritems():

            # iterate over recorded variables
            for var in group.record_variables:
                
                # numpy array 
                data_array = getattr(group, var)
                n = data_array.shape[0]

                if group_type_key == 'synapses':
                    pre_index = group.source.i
                    post_index = group.source.j
                else:
                    pre_index = np.array([np.nan]*n)
                    post_index = np.array([np.nan]*n)
                
                group_data[var] = {

                # numpy array of recorded data (recorded objects x time)
                'data':data_array,

                # index of recorded objects in brian space
                'index':group.record,

                # indices of pre and post synaptic neurons (for synapses only, nan's are entered otherwise)
                'pre_index':pre_index,
                'post_index':post_index,

                # name of brian object
                'brian_group_name':np.array([group.source.name]*n),
                
                # name of object in higher level namespace
                'group_name':np.array([group_key]*n),

                # unique trial identifier
                'trial_id': np.array([P.simulation['trial_id']]*n),

                # parameter dictionary objects
                'P':np.array([P]*n),

                # electric field magnitude
                'field_mag':np.array([P.simulation['field_mag']]*n),
                }

    return group_data



def _dict2frame(data_dict):
    '''
    '''
    data_frames = {}
    for key, val in data_dict.iteritems():

        data_frames[key] = pd.DataFrame( dict( [(k, pd.Series(list(v))) for k,v in val.iteritems()]))
    return data_frames


#     dict_copy = copy.deepcopy(data_dict)
#     for group_type_key, group_type in dict_copy.iteritems():
#         for group_key, group in group_type.iteritems():
#             for var_key, var in group.iteritems():
#                 dict_copy[group_type_key][group_key][var] = pd.Series(var.tolist())
#     for key, val in dict_copy.iteritems():
#     frame = pd.DataFrame()

def _add_to_group_data(group_data, data_dict):
    '''
    '''
    for variable_key, variable in data_dict.iteritems():
        if variable_key not in group_data:
            group_data[variable_key] = variable
            continue
        for data_key, data in variable.iteritems():
            if data_key not in group_data[variable_key]:
                group_data[variable_key][data_key] = data
            else:
                # print data_key, group_data[variable_key][data_key]
                # print data_key, data
                group_data[variable_key][data_key] = np.append(group_data[variable_key][data_key], data, axis=0)

    return group_data
