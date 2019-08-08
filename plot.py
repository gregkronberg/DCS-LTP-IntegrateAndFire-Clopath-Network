from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import copy


def _plot_all(group_dataframe, variable):
    '''
    '''
    gf = group_dataframe[variable]
    plt.figure()
    plt.plot(np.array(gf.data[gf.field_mag==0].tolist()).squeeze().T, color='black')
    plt.plot(np.array(gf.data[gf.field_mag>0].tolist()).squeeze().T, color='red')
    plt.show(block=False)

def _plot_mean(df, variable, group='1'):
    '''
    '''
    df = df[variable][group]
    control_array = np.array(df.data[df.field_mag==0].tolist()).squeeze().T
    anodal_array = np.array(df.data[df.field_mag>0].tolist()).squeeze().T
    control_mean = np.mean(control_array, axis=1)
    anodal_mean = np.mean(anodal_array, axis=1)
    control_std = np.std(control_array, axis=1)
    anodal_std = np.std(anodal_array, axis=1)
    control_sem = stats.sem(control_array, axis=1)
    anodal_sem = stats.sem(anodal_array, axis=1)
    t = np.arange(len(control_mean))
    plt.figure()
    plt.plot(control_mean, color='k')
    plt.plot(anodal_mean, color='r')
    plt.fill_between(t, control_mean-control_sem, control_mean+control_sem, color='k', alpha=0.8)
    plt.fill_between(t, anodal_mean-anodal_sem, anodal_mean+anodal_sem, color='r', alpha=0.8)
    # plt.plot(np.array(gf.data[gf.field_mag==0].tolist()).squeeze().T, color='black')
    # plt.plot(np.array(gf.data[gf.field_mag>0].tolist()).squeeze().T, color='red')
    plt.show(block=False)

def _plot_bar(df_sorted, figure_params, variable, group_space=1, bar_width=1, bar_spacing=1):
    '''
    '''
    # print 'look here:',figure_params.keys()
    fig={}
    ax={}
    n_subgroups={}
    n_traces={}
    xlim={}
    ylim={}
    for figure_key, figure_subgroups in figure_params.iteritems():
        if figure_key!='params':
            fig[figure_key], ax[figure_key] = plt.subplots()
            n_subgroups[figure_key] = len(figure_subgroups.keys()) 
            n_traces[figure_key]={}
            locations = []
            heights=[]
            colors=[]
            fig_args=[]
            xticks=[]
            xticklabels=[]
            cnt=bar_spacing
            for subgroup_key, traces in figure_subgroups.iteritems():
                if subgroup_key!='params':
                    n_traces[figure_key][subgroup_key]=len(traces.keys())
                    cnt+=group_space
                    for trace_key, params in traces.iteritems():
                        if trace_key!='params':
                            trace_args={}
                            cnt+=bar_spacing
                            locations.append(cnt)
                            xticks.append(cnt)
                            xticklabels.append(params['label'])

                            # get data and stats
                            trace_series = df_sorted[trace_key][variable]
                            data_array = (_2array(trace_series, remove_nans=True, remove_nans_axis=1)-1)*100.
                            # get stats
                            # mean across slices
                            data_mean = np.mean(data_array, axis=0)
                            #std across slices
                            data_std = np.std(data_array, axis=0)
                            # sem across slices
                            data_sem = stats.sem(data_array, axis=0)

                            heights.append(data_mean)

                            trace_args['color'] = params['color']

                            fig_args.append(trace_args)

                            colors.append(params['color'])

                            plt.errorbar(cnt, data_mean, data_sem, color=(.5,.5,.5))

            barcontainer = ax[figure_key].bar(locations, heights, width=bar_width, tick_label=xticklabels)
            xlim[figure_key] = ax[figure_key].get_xlim()
            ylim[figure_key] = ax[figure_key].get_ylim()
            # ax[figure_key].set_xticks(xticks, xticklabels,)
            # barcontainer = ax[figure_key].violinplot(locations, heights, width=bar_width, tick_label=xticklabels)
            print 'rotations:', figure_key, figure_params[figure_key]['params']
            ax[figure_key].set_xticklabels(xticklabels, fontsize
                =20, fontweight='heavy', rotation=figure_params[figure_key]['params']['rotation'])
            for bar_i, bar in enumerate(barcontainer):
                bar.set_color(colors[bar_i])

    # get ylim and xlim across all figures
    xlims=[]
    ylims=[]
    for figure_key in ylim:
        xlims.append(xlim[figure_key])
        ylims.append(ylim[figure_key])
    xlim_all = [min([temp[0] for temp in xlims]), max([temp[1] for temp in xlims])]
    ylim_all = [min([temp[0] for temp in ylims]), max([temp[1] for temp in ylims])]

    xlim={}
    ylim={}
    print figure_params.keys()
    for figure_key, axes in ax.iteritems():
        if 'ylim_all' in figure_params['params'] and figure_params['params']['ylim_all']:
            print 'setting ylim'
            ax[figure_key].set_ylim(ylim_all)
            ax[figure_key].set_xlim(xlim_all)

        
        # format figure
        ax[figure_key].spines['right'].set_visible(False)
        ax[figure_key].spines['top'].set_visible(False)
        ax[figure_key].spines['left'].set_linewidth(5)
        ax[figure_key].spines['bottom'].set_linewidth(5)
        ax[figure_key].xaxis.set_ticks_position('bottom')
        ax[figure_key].yaxis.set_ticks_position('left')
        # ax[figure_key].set_xlabel('Time (min)', fontsize=25, fontweight='heavy')
        ax[figure_key].set_ylabel('% LTP', fontsize=25, fontweight='heavy')
        xticks = np.arange(0,81, 20)
        ytickmax = ax[figure_key].get_ylim()[1]
        ytickmin = ax[figure_key].get_ylim()[0]
        yticks = np.round(np.arange(0,ytickmax, 10), decimals=0).astype(int)
        # ax[figure_key].set_xticks(xticks)
        # ax[figure_key].set_xticklabels(xticks, fontsize=20, fontweight='heavy')
        ax[figure_key].set_yticks(yticks)
        ax[figure_key].set_yticklabels(yticks, fontsize=20, fontweight='heavy')
        
        # ax[figure_key].set_ylim(ylim[figure_key])
        # ax[figure_key].set_xlim(xlim)
        plt.figure(fig[figure_key].number)
        plt.tight_layout()

    plt.show(block=False)

    return fig, ax

class Plot:
    '''
    '''
    def __init__(self,):
        '''
        '''

    def _plot_voltage(self, rec):
        '''
        '''
        self.u  = rec['nrn'].u
        linewidth=4
        colors = ['r','k','b']
        plt.figure()
        for i in range(self.u.shape[0]):
            plt.plot(rec['nrn'].t/ms, self.u[i,:]/mV, color=colors[i], linewidth=linewidth)

        plt.xlabel('Time (ms)', fontsize=20, fontweight='bold')
        plt.ylabel('Membrane potential (mV)', fontsize=20, fontweight='bold')
        plt.ylim([-72,-20])
        plt.show(block=False)


    def _plot_weights(self, rec, npre, npost):
        '''
        '''
        self.w = rec['syn'].w_clopath
        self.w = self.w.reshape(npre,npost,self.w.shape[1])
        markers = ['-','--']
        colors = ['r','k','b']
        linewidth=4
        plt.figure()
        for pre_i in range(npre-1):
            for post_i in range(npost):
                plt.plot(rec['syn'].t/ms, self.w[pre_i,post_i,:], color=colors[post_i], linestyle=markers[pre_i], linewidth=linewidth)
        plt.xlabel('Time (ms)', fontsize=20, fontweight='bold')
        plt.ylabel('Synaptic Weight (AU)', fontsize=20, fontweight='bold')
        plt.ylim([0.4,1.5])
        plt.show(block=False)
    # FIXME
    def _rate_x_location(self, rec):
        '''
        '''
        tol = .01
        integration_window = 1000
        firing_rate_filter = np.ones(integration_window)
        locations=rec['locations']

        anodal_idx = rec['nrn_spikes'].i == 0
        control_idx = rec['nrn_spikes'].i == 1
        cathodal_idx = rec['nrn_spikes'].i == 2
        anodal_spikes = rec['nrn_spikes'].t[anodal_idx]
        control_spikes = rec['nrn_spikes'].t[control_idx]
        cathodal_spikes = rec['nrn_spikes'].t[cathodal_idx]
        time = np.linspace(0, locations.shape[0]*defaultclock.dt, locations.shape[0])
        locations_unique = np.array(list(set(np.round(locations, 0))))
        print locations_unique.shape[0],time.shape[0]
        spike_array_anodal = np.zeros((locations_unique.shape[0],time.shape[0]  ))
        
        for spike_i, spike in enumerate(anodal_spikes):
            # print np.round(spike/ms,2)
            # print np.round(time[10]/ms,2)
            # time_i = np.where(np.round(time/ms,2)==np.round(spike/ms,2))[0]
            # time_i = np.where(abs(time/ms-spike/ms) < tol)
            time_i = np.argmin(abs(time/ms-spike/ms))
            location = locations[time_i]
            # print np.round(spike/ms,2), time_i, location, min(abs(time/ms-spike/ms))

            location_unique_i = np.argmin(abs(locations_unique-location))
            print location_unique_i, time_i
            # print location_unique_i, time_i
            spike_array_anodal[location_unique_i,time_i]=1

        firing_rate_array_anodal = np.zeros(spike_array_anodal.shape)
        for loc_i in range(spike_array_anodal.shape[0]):
            firing_rate_array_anodal[loc_i, :] = np.convolve(spike_array_anodal[loc_i,:], firing_rate_filter, mode='same')


        rec['time'] = time
        rec['locations_unique'] = locations_unique
        rec['anodal_spikes'] = anodal_spikes
        rec['cathodal_spikes'] =  cathodal_spikes
        rec['control_spikes'] = control_spikes

        plt.figure()
        plt.imshow(firing_rate_array_anodal, aspect='auto')
        plt.colorbar()
        plt.show(block=False)

        return spike_array_anodal

    def _time_series(self, rec, variables, indices):
        '''
        ==Args==
        -rec : dictionary of recorded StateMonitor objects
        -variables : list of variables to plot
        ==Out==
        ==Update==
        ==Comments==
        '''
        # object for dimensionless variable
        dimensionless = units.fundamentalunits.DIMENSIONLESS
        # map dimension to specific unit
        dimension_map = [(volt, mV, 'mV'), (amp,pA,'pA'), (siemens,nS,'nS'), (dimensionless,1, 'AU')]
        # iterate over variables
        for variable_i, variable in enumerate(variables):
            # iterate through recorded data types
            for rec_type, rec_data in rec.iteritems():

                if isinstance(rec_data, StateMonitor) and variable in rec_data.recorded_variables.keys():
                    figure()
                    plot_time = getattr(rec_data, 't')
                    plot_data = getattr(rec_data, variable)
                    plot_dimensions = get_dimensions(plot_data)
                    print plot_dimensions
                    plot_units = [temp[1] for temp_i, temp in enumerate(dimension_map) if plot_dimensions==get_dimensions(temp[0])][0]
                    print plot_units
                    plot(plot_time/ms, plot_data[indices[variable_i]].T/plot_units,)
                    show(block=False)

                elif isinstance(rec_data, SpikeMonitor) and variable in list(rec_data.record_variables):
                    figure()
                    plot_time = getattr(rec_data, 't')
                    plot_data = getattr(rec_data, variable)
                    plot_dimensions = get_dimensions(plot_data)
                    plot_units = [temp[1] for temp_i, temp in enumerate(dimension_map) if plot_dimensions==temp[0]][0] 
                    plot(plot_time/ms, plot_data[indices[variable_i]].T/plot_units)
                    show(block=False)