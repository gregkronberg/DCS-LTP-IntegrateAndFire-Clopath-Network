'''
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
import functions

# FIXME use timed array to turn field on and off at specific time
# choose a poissoninput group that is always paired with dcs
# use the same timedarray (corresponding row from input_timed_array)

# FIXME assign spatial variable to each neuron and use that to design connectivity
# directory and file name to store data
#====================================================================
exp_name = '.'.join(__name__.split('analysis_')[1:])
group_data_directory = 'Datatemp/'+exp_name+'/'
group_data_filename = exp_name+'_data.pkl'
group_data_filename_train = exp_name+'_data_train.pkl'
group_data_filename_test = exp_name+'_data_test.pkl'

# dictionary for group data over multiple trials
train_group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename_train, df=True)
test_group_df = analysis._load_group_data(directory=group_data_directory, file_name=group_data_filename_test, df=True)

# print train_group_df.keys()
# df_w_train = train_group_df[train_group_df.variable=='w_clopath']
# df_u_train = train_group_df[train_group_df.variable=='u']
# df_w_test = test_group_df[train_group_df.variable=='w_clopath']
# df_u_test = test_group_df[train_group_df.variable=='u']

# df_w_train_FF = df_w_train[df_w_train.group_name=='FF_train']
# df_w_train_EE = df_w_train[df_w_train.group_name=='EE']
# df_u_train_E = df_u_train[df_u_train.group_name=='E']
# df_u_train_I = df_u_train[df_u_train.group_name=='I']
# df_u_test_E = df_u_test[df_u_train.group_name=='E']
# df_u_test_I = df_u_test[df_u_train.group_name=='I']

# w_train_FF = functions._2array(df_w_train[df_w_train.group_name=='FF_train'].data)
# w_train_EE = functions._2array(df_w_train[df_w_train.group_name=='EE'].data)
# u_train_E = functions._2array(df_u_train[df_u_train.group_name=='E'].data)
# u_train_I = functions._2array(df_u_train[df_u_train.group_name=='I'].data)
# u_test_E = functions._2array(df_u_test[df_u_train.group_name=='E'].data)
# u_test_I =functions._2array(df_u_test[df_u_train.group_name=='I'].data)

# FIXME GET FIRING RATE OF EACH NEURON DURING TRAINING AND TEST EPOCHS