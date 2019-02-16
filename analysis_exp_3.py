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
import os
import functions
from collections import OrderedDict

exp_name = '.'.join(__name__.split('analysis_')[1:])
print exp_name
group_data_directory = 'Data/'+exp_name+'/'
group_data_filename = exp_name+'_data.pkl'
figure_directory = 'Figures/'+exp_name+'/'

# check if folder exists with experiment name
if os.path.isdir(figure_directory) is False:
    print 'making new directory to save data'
    os.mkdir(figure_directory)

# dictionary for group data over multiple trials
df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename, df=True)

print df.keys()

df_w = df[df.variable=='w_clopath']

df_w['ltp_final'] = df_w.apply(lambda row: analysis._build_ltp_final(row), axis=1)

# print df_w.ltp_final

conditions = ['field_polarity', 'pre_brian_group_name', 'post_brian_group_name']

constraints_spec = OrderedDict(
    []
    ) 
# constraint applied to all conditions. this gets overwritten by specific constraints
constraints_all = OrderedDict([
    ('variable', ['==','w_clopath'])
    ])

df_w_sorted = functions._sortdf(df=df_w, conditions=conditions, constraints_all=constraints_all, constraints_spec=constraints_spec)

figure_params = {
    'params':{
        'ylim_all':True,

    },
    # figure name
    'weak_associative':{
        'params':{
        'rotation':0

        },
        # subgroup of traces
        'control':{
            'params':{

            },
            # individual traces
            ('control', 'inputs_2', 'neurons_3'): {
                # trace parameters
                'color':'k',
                'label': '5Hz',
                'location':1
            },

            ('control', 'inputs_2', 'neurons_2'):{
                'color':'k',
                'label':'5Hz\n+TBS',
                'location':2
            }
        },

        'anodal':{
            'params':{

            },
            # individual traces
            ('anodal', 'inputs_2', 'neurons_3'): {
                # trace parameters
                'color':'r',
                'label':'5Hz',
                'location':3

            },
            
            ('anodal', 'inputs_2', 'neurons_2'):{
                'color':'r',
                'label':'5Hz\n+TBS',
                'location':4
                }
            }
        },

    'specificity':{
            'params':{
            'rotation':45

            },
        # subgroup of traces
        'control':{
            'params':{

            },
            # individual traces
            ('control', 'inputs_1', 'neurons_1'): {
                # trace parameters
                'color':'k',
                'label': 'TBS',
                'location':2
            },

            ('control', 'inputs_3', 'neurons_1'):{
                'color':'k',
                'label':'Inactive',
                'location':1
            }
        },

        'anodal':{
            'params':{

            },
            # individual traces
            ('anodal', 'inputs_3', 'neurons_1'): {
                # trace parameters
                'color':'r',
                'label':'Inactive',
                'location':3

            },
            
            ('anodal', 'inputs_1', 'neurons_1'):{
                'color':'r',
                'label':'TBS',
                'location':4
                }
            }
        }
        }

print 'fig_params:',figure_params.keys()
bar_figs, bar_axes = functions._plot_bar(df_sorted=df_w_sorted, figure_params=figure_params, variable='ltp_final')

for fig_key, fig in bar_figs.iteritems():
    fname = figure_directory+'bar_'+str(fig_key)+'.png'
    fig.savefig(fname, format='png', dpi=350)