class LIFnetworks(object):
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''
        pass

    def _add_eq(self, *list_args):

        '''
        '''
        equation  = ''
        for eq in list_args:
            equation = equation + '\n' + eq 

        return equation 



class AdexBonoClopath:

    def __init__(self, ):
        '''
        '''

        self.neuron = '''

            du/dt = int(not_refractory)*I_total/C  + I_reset  + ((t-lastspike)>spike_hold_time2)*(1-int(not_refractory))*I_total/C :volt

            I_total = (I_L + I_syn + I_exp + I_field + I_input - w_adapt + z_after ) : amp

            I_reset = ((t-lastspike)<spike_hold_time2)*((t-lastspike)>spike_hold_time)*(1-int(not_refractory))*((u_reset-u)/t_reset + z_after/C) : volt/second     

            I_L = -g_L*(u - E_L) : amp

            I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

            dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

            dw_adapt/dt = a_adapt*(u-E_L)/t_w_adapt - w_adapt/t_w_adapt : amp

            dz_after/dt = -z_after/t_z_after : amp 

            # synaptic
            #=======================================
            # ampa
            #``````````````
                dg_ampa/dt = -g_ampa/t_ampa : siemens 
                I_ampa = -g_ampa*(u-E_ampa) : amp

            # nmda
            #`````````````````
                dg_nmda/dt = -g_nmda/t_nmda : siemens 
                B =  1/(1 + exp(-0.062*u/mV)/3.57) : 1 
                I_nmda = -g_nmda*B*(u-E_nmda) : amp

            # gaba
            #````````````````````````````
                dg_gaba/dt = -g_gaba/t_gaba : siemens
                I_gaba = -g_gaba*(u-E_gaba) : amp

            # total synaptic current
            #```````````````````````````````
            I_syn = I_ampa + I_nmda + I_gaba: amp

            # clopath
            #```````````````````
            # low threshold filtered membrane potential
                du_lowpass1/dt = (u-u_lowpass1)/tau_lowpass1 : volt 

            # high threshold filtered membrane potential
                du_lowpass2/dt = (u-u_lowpass2)/tau_lowpass2 : volt     

            # homeostatic term
                du_homeo/dt = (u-E_L-u_homeo)/tau_homeo : volt       

            # LTP voltage dependence
                LTP_u = (u_lowpass2/mV - theta_low/mV)*int((u_lowpass2/mV - theta_low/mV) > 0)*(u/mV-theta_high/mV)*int((u/mV-theta_high/mV) >0)  : 1

            # LTD voltage dependence
                LTD_u = (u_lowpass1/mV - theta_low/mV)*int((u_lowpass1/mV - theta_low/mV) > 0)  : 1

            # homeostatic depression amplitude
                #``````````````````````````````````
                A_LTD_homeo = (1-include_homeostatic)*A_LTD + (include_homeostatic)*A_LTD*(u_homeo**2/v_target) : 1  
            
            # parameters
            #```````````````
            # I_input : amp
            I_field : amp
            # I_syn : amp
            # I_after : amp
            # C : farad
            # g_L : siemens
            # delta_T : volt 
            # t_V_T : second
            # a_adapt : siemens
            # t_w_adapt : second
            # t_z_after : second
            # u_reset : volt
            # b_adapt : amp
            # V_Tmax : volt
            # V_Trest:volt
            # E_L : volt
        '''
        # voltage rest
        #``````````````````````````````
        self.neuron_reset ='''
            z_after = I_after 
            u = int(hold_spike)*u_hold + int(1-hold_spike)*(u_reset + dt*I_after/C)
            V_T = V_Tmax 
            w_adapt += b_adapt    
        '''
        # ampa synapses
        #'````````````````````````````````
        self.syn_ampa = '''
            # dg_ampa/dt = -g_ampa/t_ampa : siemens 
            # I_ampa = -g_ampa*(u_post-E_ampa) : amp
            
            w_ampa :1
            # g_max_ampa : siemens
            # t_ampa : second
            # E_ampa : volt
            '''

        self.syn_gaba = '''
                w_gaba:1
                '''

        # inhibition without presynaptic adaptation (multiply each term by A to include adaptation)
        self.syn_gaba_pre = '''
            g_gaba += int(update_gaba_online)*w_vogels*g_max_gaba + int(1-update_gaba_online)*w_gaba*g_max_ampa 
            
                '''

        self.syn_ampa_pre = '''
            g_ampa += int(update_ampa_online)*w_clopath*g_max_ampa*A + int(1-update_ampa_online)*w_ampa*g_max_ampa*A 
            '''
        self.syn_ampa_pre_nonadapt = '''
            g_ampa += int(update_ampa_online)*w_clopath*g_max_ampa + int(1-update_ampa_online)*w_ampa*g_max_ampa 
            '''

        self.syn_nmda_pre = '''
            g_nmda += int(update_nmda_online)*w_clopath*g_max_nmda*A + int(1-update_nmda_online)*w_nmda*g_max_nmda*A 
            '''
        self.syn_nmda_pre_nonadapt = '''
            g_nmda += int(update_nmda_online)*w_clopath*g_max_nmda + int(1-update_nmda_online)*w_nmda*g_max_nmda
            '''
        # nmda synapses
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_nmda = '''
            w_nmda:1
            # dg_nmda/dt = -g_nmda/t_nmda : siemens 
            # B =  1/(1 + exp(-0.062*u_post/mV)/3.57) : 1 
            # I_nmda = -g_nmda*B*(u_post-E_nmda) : amp
        
            '''

        # short term plasticity
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_stp = '''

            dF/dt = (1-F)/t_F : 1 
            dD1/dt = (1-D1)/t_D1 : 1 
            dD2/dt = (1-D2)/t_D2 : 1 
            dD3/dt = (1-D3)/t_D3 : 1 
            A = F*D1*D2*D3 : 1
            '''

        self.syn_stp_pre = '''
            F += f 
            D1 *= d1
            D2 *= d2
            D3 *= d3 
            '''

        # vogels inhibitory plasticity rule
        #````````````````````````````````````````````````````````````````````
        self.syn_vogels = '''
            w_vogels:1
            dApre_vogels/dt = -Apre_vogels/tau_vogels : 1
            dApost_vogels/dt = -Apost_vogels/tau_vogels : 1
        '''

        self.syn_vogels_pre = '''
            Apre_vogels += 1
            w_vogels = clip(w_vogels+(Apost_vogels-alpha_vogels)*eta_vogels, 0, w_max_vogels)
            '''

        self.syn_vogels_post = '''
            Apost_vogels += 1
            w_vogels = clip(w_vogels+Apre_vogels*eta_vogels, 0, w_max_vogels)
        '''

        # clopath plasticity rule
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_clopath = '''      
        
            # lowpass presynaptic variable
                dx_trace/dt = -x_trace/tau_x : 1                          

            # clopath rule for potentiation (depression implented with on_pre)

            # dw_clopath/dt = int(w_clopath<w_max_clopath)*A_LTP*x_trace*LTP_u_post :1
            
            dw_clopath/dt = saturated*(w_max_clopath-w_clopath)/dt + (1-saturated)*A_LTP*x_trace*LTP_u_post : 1

            saturated = int((w_clopath+A_LTP*x_trace*LTP_u_post*dt)>w_max_clopath) : 1 # indicates that next weight update brings synapse to saturation
            '''

        self.syn_clopath_pre = '''

            w_minus = A_LTD_homeo_post*LTD_u_post
            
            w_clopath = clip(w_clopath-w_minus, 0, w_max_clopath)  # apply LTD
            # w_clopath = w_clopath-w_minus  # apply LTD without bound

            x_trace += dt*x_reset/tau_x  # update presynaptic trace with each input
            '''

        self.synapse_e = _add_eq(self.syn_ampa, self.syn_nmda, self.syn_stp, self.syn_clopath)

        self.synapse_i = _add_eq(self.syn_gaba, self.syn_vogels)

        self.synapse_e_pre = _add_eq(self.syn_ampa_pre, self.syn_nmda_pre, self.syn_stp_pre, self.syn_clopath_pre)

        self.synapse_e_pre_nonadapt = _add_eq(self.syn_ampa_pre_nonadapt, self.syn_nmda_pre_nonadapt, self.syn_stp_pre, self.syn_clopath_pre)

        self.synapse_i_pre = _add_eq(self.syn_gaba_pre, self.syn_vogels_pre)

        self.synapse_i_post = _add_eq(self.syn_vogels_post)   

class AdexLitwinKumar2014:

    def __init__(self, ):
        '''
        '''

        self.neuron = '''
            # voltage
            ################################################################
            # du/dt = I_total/C  :volt
            du/dt = int(u_max<(u+dt*I_total/C) and u!=u_max)*(u_max-u)/dt + int(u_max>=u+dt*I_total/C)*I_total/C  :volt
            # du/dt = int(u_max>=u+dt*I_total/C)*I_total/C  :volt
            # u =clip(u, -80*mV, 20*mV) :volt

            du_test/dt = int(u>u_max)*(u_max - u_test)/dt + (1-int(u>u_max))*(u - u_test)/dt : volt

            # u = clip(u_temp, -100, u_max)
            # u_clopath=u:volt

            I_total = (I_L + I_syn + I_exp + I_field + I_input - w_adapt + z_after ) : amp   

            I_L = -g_L*(u - E_L) : amp

            # I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

            I_exp = clip(g_L*delta_T*exp((u - V_T)/delta_T), -50*nA, 50*nA) : amp

            dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

            dw_adapt/dt = a_adapt*(u-E_L)/t_w_adapt - w_adapt/t_w_adapt : amp

            dz_after/dt = -z_after/t_z_after : amp 

            # synaptic
            ###############################################################
            # ampa
            #-------
            dg_ampa/dt = -g_ampa/t_ampa : siemens 
            I_ampa = -g_ampa*(u-E_ampa) : amp

            # nmda
            #--------
            dg_nmda/dt = -g_nmda/t_nmda : siemens 
            B =  1/(1 + exp(-0.062*u/mV)/3.57) : 1 
            I_nmda = -g_nmda*B*(u-E_nmda) : amp

            # gaba
            #-------
            dg_gaba/dt = -g_gaba/t_gaba : siemens
            I_gaba = -g_gaba*(u-E_gaba) : amp

            # total synaptic current
            #-------------------------
            I_syn = I_ampa + I_nmda + I_gaba: amp

            # clopath
            ###############################################################
            # low threshold filtered membrane potential
            #---------------------------------------------
            du_lowpass1/dt = (u_test-u_lowpass1)/tau_lowpass1 : volt 

            # high threshold filtered membrane potential
            #--------------------------------------------
            du_lowpass2/dt = (u_test-u_lowpass2)/tau_lowpass2 : volt     

            # homeostatic term
            #-------------------
            du_homeo/dt = (u_test-E_L-u_homeo)/tau_homeo : volt       

            # LTP voltage dependence
            #-------------------------
            LTP_u = (u_lowpass2/mV - theta_low/mV)*int((u_lowpass2/mV - theta_low/mV) > 0)*(u_test/mV-theta_high/mV)*int((u_test/mV-theta_high/mV) >0)  : 1

            # LTD voltage dependence
            #-----------------------
            LTD_u = (u_lowpass1/mV - theta_low/mV)*int((u_lowpass1/mV - theta_low/mV) > 0)  : 1

            # homeostatic depression amplitude
            #-----------------------------------
            A_LTD_homeo = (1-include_homeostatic)*A_LTD + (include_homeostatic)*A_LTD*(u_homeo**2/v_target) : 1  

            # total excitatory (clopath) weights
            #-----------------------------------
            w_clopath_total = w_clopath_total_recurrent + w_clopath_total_feedforward :1
            
            # parameters
            ################################################################
            w_clopath_total_recurrent:1
            w_clopath_total_feedforward:1
            N_syn_total_recurrent:1
            N_syn_total_feedforward:1
            I_input : amp
            I_field : amp
            # I_syn : amp
            # I_after : amp
            # C : farad
            # g_L : siemens
            # delta_T : volt 
            # t_V_T : second
            # a_adapt : siemens
            # t_w_adapt : second
            # t_z_after : second
            # u_reset : volt
            # b_adapt : amp
            # V_Tmax : volt
            # V_Trest:volt
            # E_L : volt
        '''
        self.neuron_reset ='''
            z_after = I_after 
            u = u_reset
            u_test = u_max
            V_T = V_Tmax 
            w_adapt += b_adapt   

            u_lowpass1 += (u_spike_clopath-u_lowpass1)/tau_lowpass1*ms
            u_lowpass2 += (u_spike_clopath-u_lowpass2)/tau_lowpass2*ms
            u_homeo += (u_spike_clopath-E_L-u_homeo)/tau_homeo*ms   

        '''

        self.syn_ampa = '''
            w_ampa :1
            '''
        self.syn_nmda = '''
            w_nmda:1
        
            '''
        self.syn_gaba = '''
            w_gaba:1
            '''

        # inhibition without presynaptic adaptation (multiply each term by A to include adaptation)
        self.syn_gaba_pre = '''
            g_gaba += int(update_gaba_online)*w_vogels*g_max_gaba + int(1-update_gaba_online)*w_gaba*g_max_gaba
            '''

        self.syn_ampa_pre = '''
            g_ampa += int(update_ampa_online)*w_clopath*g_max_ampa*A + int(1-update_ampa_online)*w_ampa*g_max_ampa*A 
            '''
        self.syn_ampa_pre_nonadapt = '''
            g_ampa += update_ampa_online*w_clopath*g_max_ampa + (1-update_ampa_online)*w_ampa*g_max_ampa 
            '''

        self.syn_nmda_pre = '''
            g_nmda += int(update_nmda_online)*w_clopath*g_max_nmda*A + int(1-update_nmda_online)*w_nmda*g_max_nmda*A 
            '''
        self.syn_nmda_pre_nonadapt = '''
            g_nmda += int(update_nmda_online)*w_clopath*g_max_nmda + int(1-update_nmda_online)*w_nmda*g_max_nmda

            '''

        # short term plasticity
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_stp = '''

            dF/dt = (1-F)/t_F : 1 
            dD1/dt = (1-D1)/t_D1 : 1 
            dD2/dt = (1-D2)/t_D2 : 1 
            dD3/dt = (1-D3)/t_D3 : 1 
            A = F*D1*D2*D3 : 1
            '''

        self.syn_stp_pre = '''
            F += f 
            D1 *= d1
            D2 *= d2
            D3 *= d3 
            '''

        # vogels inhibitory plasticity rule
        #````````````````````````````````````````````````````````````````````
        self.syn_vogels = '''
            w_vogels:1
            dApre_vogels/dt = -Apre_vogels/tau_vogels : 1 (event-driven)
            dApost_vogels/dt = -Apost_vogels/tau_vogels : 1 (event-driven)
        '''

        self.syn_vogels_pre = '''
            Apre_vogels += 1
            w_vogels = clip(w_vogels+(Apost_vogels-alpha_vogels)*eta_vogels, w_min_vogels, w_max_vogels)

            '''

        self.syn_vogels_post = '''
            Apost_vogels += 1
            w_vogels = clip(w_vogels+Apre_vogels*eta_vogels, w_min_vogels, w_max_vogels)
        '''

        # clopath plasticity rule
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_clopath = '''      
        
            # lowpass presynaptic variable
            #--------------------------------
            dx_trace/dt = -x_trace/tau_x : 1 (event-driven)                          

            # clopath rule for potentiation (depression implented with on_pre)

            # dw_clopath/dt = int(w_clopath<w_max_clopath)*A_LTP*x_trace*LTP_u_post :1
            
            dw_clopath/dt = saturated*(w_max_clopath-w_clopath)/dt+(1-saturated)*A_LTP*x_trace*LTP_u_post : 1 (clock-driven)

            w_clopath_total_post = w_clopath :1 (summed)

            saturated = int((w_clopath+A_LTP*x_trace*LTP_u_post*dt)>w_max_clopath) : 1 # indicates that next weight update brings synapse to saturation
            '''

        self.syn_clopath_feedforward = '''      
        
            # lowpass presynaptic variable
            #--------------------------------
            dx_trace/dt = -x_trace/tau_x : 1 (event-driven)                          

            # clopath rule for potentiation (depression implented with on_pre)

            # dw_clopath/dt = int(w_clopath<w_max_clopath)*A_LTP*x_trace*LTP_u_post :1
            
            dw_clopath/dt = saturated*(w_max_clopath-w_clopath)/dt+(1-saturated)*A_LTP*x_trace*LTP_u_post - include_normalization*normalize*w_clopath_norm_factor/t_norm_w_clopath : 1 (clock-driven)

            saturated = int((w_clopath+A_LTP*x_trace*LTP_u_post*dt)>w_max_clopath) : 1 # indicates that next weight update brings synapse to saturation

            # normalize synapses
            #---------------------
            w_clopath_total_feedforward_post = w_clopath-w_init_clopath :1 (summed)

            w_clopath_norm_factor = w_clopath_total_feedforward_post/N_incoming :1

            normalize=1:1

            # N_syn_total_feedforward_post = 1:1(summed)

            # normalize = int((t%t_norm_w_clopath) ==0*ms) :1 
            
            '''

        self.syn_clopath_recurrent = '''      
        
            # lowpass presynaptic variable
            #--------------------------------
            dx_trace/dt = -x_trace/tau_x : 1 (event-driven)                          

            # clopath rule for potentiation (depression implented with on_pre)

            # dw_clopath/dt = int(w_clopath<w_max_clopath)*A_LTP*x_trace*LTP_u_post :1
            
            dw_clopath/dt = saturated*(w_max_clopath-w_clopath)/dt+(1-saturated)*A_LTP*x_trace*LTP_u_post - include_normalization*normalize*w_clopath_norm_factor/t_norm_w_clopath : 1 (clock-driven)

            # dw_clopath/dt = saturated*(w_max_clopath-w_clopath)/dt+(1-saturated)*A_LTP*x_trace*LTP_u_post : 1 (clock-driven)

            saturated = int((w_clopath+A_LTP*x_trace*LTP_u_post*dt)>w_max_clopath) : 1 # indicates that next weight update brings synapse to saturation

            # normalize synapses
            #---------------------

            w_clopath_total_recurrent_post = w_clopath-w_init_clopath :1 (summed)

            w_clopath_norm_factor = w_clopath_total_recurrent_post/(N_incoming) :1

            normalize=1:1

            # N_syn_total_recurrent_post = 1:1(summed)

            # w_clopath_norm_factor = w_clopath_total_recurrent_post/N_incoming :1

            # normalize = int((t%t_norm_w_clopath) < dt) :1 

            
            '''

        self.syn_clopath_pre = '''

            w_minus = A_LTD_homeo_post*LTD_u_post
            
            w_clopath = clip(w_clopath-w_minus, w_min_clopath, w_max_clopath)  # apply LTD


            x_trace += dt*x_reset/tau_x  # update presynaptic trace with each input
            '''



        self.synapse_e = _add_eq(self.syn_ampa, self.syn_clopath)

        self.synapse_e_feedforward = _add_eq(self.syn_ampa, self.syn_clopath_feedforward)
        
        self.synapse_e_recurrent = _add_eq(self.syn_ampa, self.syn_clopath_recurrent)

        self.synapse_i = _add_eq(self.syn_gaba, self.syn_vogels)

        self.synapse_e_pre = _add_eq(self.syn_ampa_pre, self.syn_clopath_pre)

        self.synapse_e_pre_nonadapt = _add_eq(self.syn_ampa_pre_nonadapt, self.syn_clopath_pre)

        self.synapse_i_pre = _add_eq(self.syn_gaba_pre, self.syn_vogels_pre)

        self.synapse_i_post = _add_eq(self.syn_vogels_post)

class AdexLitwinKumar2014cython:

    def __init__(self, ):
        '''
        '''

        self.neuron = '''
            # voltage
            ################################################################
            du/dt = I_total/C  :volt

            du_test/dt = (u - u_test)/t_u_test : volt

            # u = clip(u_temp, -100, u_max)
            # u_clopath=u:volt

            I_total = (I_L + I_syn + I_exp + I_field + I_input - w_adapt + z_after ) : amp   

            I_L = -g_L*(u - E_L) : amp

            I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

            dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

            dw_adapt/dt = a_adapt*(u-E_L)/t_w_adapt - w_adapt/t_w_adapt : amp

            dz_after/dt = -z_after/t_z_after : amp 

            # synaptic
            ###############################################################
            # ampa
            #-------
            dg_ampa/dt = -g_ampa/t_ampa : siemens 
            I_ampa = -g_ampa*(u-E_ampa) : amp

            # nmda
            #--------
            dg_nmda/dt = -g_nmda/t_nmda : siemens 
            B =  1/(1 + exp(-0.062*u/mV)/3.57) : 1 
            I_nmda = -g_nmda*B*(u-E_nmda) : amp

            # gaba
            #-------
            dg_gaba/dt = -g_gaba/t_gaba : siemens
            I_gaba = -g_gaba*(u-E_gaba) : amp

            # total synaptic current
            #-------------------------
            I_syn = I_ampa + I_nmda + I_gaba: amp

            # clopath
            ###############################################################
            # low threshold filtered membrane potential
            #---------------------------------------------
            du_lowpass1/dt = (u-u_lowpass1)/tau_lowpass1 : volt 

            # high threshold filtered membrane potential
            #--------------------------------------------
            du_lowpass2/dt = (u-u_lowpass2)/tau_lowpass2 : volt     

            # homeostatic term
            #-------------------
            du_homeo/dt = (u-E_L-u_homeo)/tau_homeo : volt       

            # LTP voltage dependence
            #-------------------------
            LTP_u = (u_lowpass2/mV - theta_low/mV)*int((u_lowpass2/mV - theta_low/mV) > 0)*(u_test/mV-theta_high/mV)*int((u_test/mV-theta_high/mV) >0)  : 1

            # LTD voltage dependence
            #-----------------------
            LTD_u = (u_lowpass1/mV - theta_low/mV)*int((u_lowpass1/mV - theta_low/mV) > 0)  : 1

            # homeostatic depression amplitude
            #-----------------------------------
            A_LTD_homeo = (1-include_homeostatic)*A_LTD + (include_homeostatic)*A_LTD*(u_homeo**2/v_target) : 1  
            
            # parameters
            ################################################################
            I_input : amp
            I_field : amp
            # I_syn : amp
            # I_after : amp
            # C : farad
            # g_L : siemens
            # delta_T : volt 
            # t_V_T : second
            # a_adapt : siemens
            # t_w_adapt : second
            # t_z_after : second
            # u_reset : volt
            # b_adapt : amp
            # V_Tmax : volt
            # V_Trest:volt
            # E_L : volt
        '''
        self.neuron_reset ='''
            z_after = I_after 
            u = u_reset
            u_test = u_max
            V_T = V_Tmax 
            w_adapt += b_adapt   

            u_lowpass1 += (u_spike_clopath-u_lowpass1)/tau_lowpass1*ms
            u_lowpass2 += (u_spike_clopath-u_lowpass2)/tau_lowpass2*ms
            u_homeo += (u_spike_clopath-E_L-u_homeo)/tau_homeo*ms   

        '''

        self.syn_ampa = '''
            w_ampa :1
            '''
        self.syn_nmda = '''
            w_nmda:1
        
            '''
        self.syn_gaba = '''
            w_gaba:1
            '''

        # inhibition without presynaptic adaptation (multiply each term by A to include adaptation)
        self.syn_gaba_pre = '''
            g_gaba += (update_gaba_online)*w_vogels*g_max_gaba + (1-update_gaba_online)*w_gaba*g_max_gaba # cython friendly
            '''

        self.syn_ampa_pre = '''
            g_ampa += (update_ampa_online)*w_clopath*g_max_ampa*A + (1-update_ampa_online)*w_ampa*g_max_ampa*A # cython friendly
            '''
        self.syn_ampa_pre_nonadapt = '''
            g_ampa += update_ampa_online*w_clopath*g_max_ampa + (1-update_ampa_online)*w_ampa*g_max_ampa 
            '''

        self.syn_nmda_pre = '''
            g_nmda += (update_nmda_online)*w_clopath*g_max_nmda*A + (1-update_nmda_online)*w_nmda*g_max_nmda*A # cython friendly
            '''
        self.syn_nmda_pre_nonadapt = '''
            g_nmda += (update_nmda_online)*w_clopath*g_max_nmda + (1-update_nmda_online)*w_nmda*g_max_nmda # cython friendly
            '''

        # short term plasticity
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_stp = '''

            dF/dt = (1-F)/t_F : 1 
            dD1/dt = (1-D1)/t_D1 : 1 
            dD2/dt = (1-D2)/t_D2 : 1 
            dD3/dt = (1-D3)/t_D3 : 1 
            A = F*D1*D2*D3 : 1
            '''

        self.syn_stp_pre = '''
            F += f 
            D1 *= d1
            D2 *= d2
            D3 *= d3 
            '''

        # vogels inhibitory plasticity rule
        #````````````````````````````````````````````````````````````````````
        self.syn_vogels = '''
            w_vogels:1
            dApre_vogels/dt = -Apre_vogels/tau_vogels : 1 (event-driven)
            dApost_vogels/dt = -Apost_vogels/tau_vogels : 1 (event-driven)
        '''

        self.syn_vogels_pre = '''
            Apre_vogels += 1
            w_vogels = w_vogels+(Apost_vogels-alpha_vogels)*eta_vogels # no clip, cython friendly
            '''

        self.syn_vogels_post = '''
            Apost_vogels += 1

            w_vogels = w_vogels+Apre_vogels*eta_vogels # no clip, cython friendly
        '''

        # clopath plasticity rule
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_clopath = '''      
        
            # lowpass presynaptic variable
            #--------------------------------
            dx_trace/dt = -x_trace/tau_x : 1 (event-driven)                          

            dw_clopath/dt = A_LTP*x_trace*LTP_u_post : 1 (clock-driven)

            '''

        self.syn_clopath_pre = '''

            w_minus = A_LTD_homeo_post*LTD_u_post
            
            w_clopath = w_clopath-w_minus  # apply LTD, no clip

            x_trace += dt*x_reset/tau_x  # update presynaptic trace with each input
            '''

        self.synapse_e = _add_eq(self.syn_ampa, self.syn_clopath)

        self.synapse_i = _add_eq(self.syn_gaba, self.syn_vogels)

        self.synapse_e_pre = _add_eq(self.syn_ampa_pre, self.syn_clopath_pre)

        self.synapse_e_pre_nonadapt = _add_eq(self.syn_ampa_pre_nonadapt, self.syn_clopath_pre)

        self.synapse_i_pre = _add_eq(self.syn_gaba_pre, self.syn_vogels_pre)

        self.synapse_i_post = _add_eq(self.syn_vogels_post)
     
class Zenke2015:
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''

        self.neuron ='''
            # subthreshold membrane voltage
            #-------------------------------
            du/dt = (E_L-u)/t_m + g_exc*(E_exc-u)/t_m + (g_gaba + g_a + g_b)*(E_gaba-u)/t_m : volt

            # threshold adaptation
            #------------------------
            du_thresh/dt = (u_thresh_rest-u_thresh)/t_u_thresh : volt

            # adaptation currents
            #----------------------
            dg_a/dt = -g_a/t_a :1
            dg_b/dt = -g_b/t_b :1

            # excitatory synaptic conductance
            #--------------------------------
            dg_ampa/dt = -g_ampa/t_ampa : 1
            dg_nmda/dt = (g_ampa-g_nmda)/t_nmda :1
            g_exc = alpha*g_ampa + (1-alpha)*g_nmda :1

            # inhibitory synaptic conductance
            #--------------------------------
            dg_gaba/dt = -g_gaba/t_gaba : 1

            # neuron specific plasticity traces
            #-----------------------------------
            dz_plus/dt = -z_plus/t_plus : 1 
            dz_minus/dt = -z_minus/t_minus : 1 
            dz_slow/dt = -z_slow/t_slow : 1 
            dz_istdp/dt = -z_istdp/t_istdp : 1 

            # short term plasticity
            #-----------------------------
            dx_stp/dt = (1-x_stp)/t_d  : 1 
            du_stp/dt = (U_stp-u_stp)/t_f : 1 

            '''

        self.neuron_reset='''
            u_thresh=u_thresh_reset
            u = E_L
            g_a += Delta_a # adaptation 
            g_b += Delta_b # adaptation
            z_plus += 1
            z_slow += 1
            z_minus += 1
            z_istdp +=1
            x_stp += -u_stp*x_stp
            u_stp += U_stp*(1-u_stp)
            '''

        self.neuron_poisson_traces='''
            rates=E_input_timed_array(t,i):Hz

            # neuron specific plasticity traces
            #-----------------------------------
            dz_plus/dt = -z_plus/t_plus : 1 
            dz_minus/dt = -z_minus/t_minus : 1 
            dz_slow/dt = -z_slow/t_slow : 1 

            # short term plasticity
            #-----------------------------
            dx_stp/dt = (1-x_stp)/t_d  : 1 
            du_stp/dt = (U_stp-u_stp)/t_f : 1 
            '''

        self.neuron_poisson_traces_reset='''
            z_plus += 1
            z_slow += 1
            z_minus += 1
            x_stp += -u_stp*x_stp
            u_stp += U_stp*(1-u_stp)
            '''

        self.neuron_global_rate_shared='''
            # G:1 #(linked)
            H:1 (linked)
            G = H-gamma:1 
            '''
        
        self.neuron_global_rate='''
            dH/dt=-H/t_H : 1 
            G = H-gamma:1 
            # gamma= 0:1
            '''
        
        self.synapse_global_rate='''
            
            '''
        self.synapse_global_rate_pre='''
            H += 1./(N_incoming)
            '''
        # self.synapse_global_rate_pre='''
        #     H += dt/(N_incoming*second)
        #     '''
        # self.synapse_global_rate_pre='''
        #     H += dt/(second)
        '''

        # self.synapse_stp='''
        #     dx_stp/dt = (1-x_stp)/t_d  : 1 (event-driven)
        #     du_stp/dt = (U_stp-u_stp)/t_f : 1 (event-driven)
            
        #     '''  

        # self.synapse_stp_pre='''
        #     x_stp += -u_stp*x_stp
        #     u_stp += U_stp*(1-u_stp)
        #     '''

        self.synapse_ampa = '''
            # w_ampa :1
            '''

        self.synapse_gaba = '''
            # w_gaba:1
            '''

        self.synapse_ampa_pre='''
            g_ampa += x_stp_pre*u_stp_pre*w_exc_plastic
            '''

        self.synapse_ampa_pre_static='''
            g_ampa += x_stp_pre*u_stp_pre*w_exc_plastic_init
            '''
        
        self.synapse_gaba_pre='''
            g_gaba += w_inh_plastic
            '''

        self.synapse_triplet_stdp='''
            w_exc_plastic :1 
            w_cons:1
            '''

        # continuous update for synaptic consolidation (very slow)
        self.synapse_consolidation='''
            dw_cons/dt = -w_cons/t_w_cons + w_exc_plastic/t_w_cons - P*w_cons*(w_P/2 - w_cons)*(w_P-w_cons)/t_w_cons : 1 (clock-driven)
            '''
        # event for updating consolidation variable (only update every consolidation_dt)
        self.neuron_consolidation_event='(t%consolidation_dt)<dt'
        
        self.synapse_consolidate_on_event=''' 
            w_cons = clip(w_cons-consolidation_dt*w_cons/t_w_cons +  consolidation_dt*w_exc_plastic/t_w_cons - consolidation_dt*P*w_cons*(w_P/2 - w_cons)*(w_P-w_cons)/t_w_cons, w_exc_plastic_min, w_exc_plastic_max )
            '''
        # self.synapse_consolidate_on_event='''
        #     w_cons += -consolidation_dt*w_cons/t_w_cons +  consolidation_dt*w_exc_plastic/t_w_cons - -consolidation_dt*P*w_cons*(w_P/2 - w_cons)*(w_P-w_cons)/t_w_cons 
        #     '''

        # self.synapse_triplet_stdp_post='''
        #     w_exc_plastic += A*z_plus*z_slow - Beta*(w_exc_plastic-w_consolidation)*z_minus**3
        #     '''

        self.synapse_triplet_stdp_post='''
            w_exc_plastic = clip(w_exc_plastic + A*z_plus_pre*z_slow_post - Beta*(w_exc_plastic-w_cons)*z_minus_post**3, w_exc_plastic_min, w_exc_plastic_max)
            '''
            
        # self.synapse_triplet_stdp_pre='''
        #     w_exc_plastic += -(B*z_minus-delta)
        #     '''
        self.synapse_triplet_stdp_pre='''
            w_exc_plastic = clip(w_exc_plastic -(B*z_minus_post-delta), w_exc_plastic_min, w_exc_plastic_max)
            
            '''

        self.synapse_istdp='''
            w_inh_plastic :1
            # dzpre_istdp/dt = -zpre_istdp/t_istdp : 1 (event-driven)
            # dzpost_istdp/dt = -zpost_istdp/t_istdp : 1 (event-driven)
            '''

        self.synapse_istdp_post='''
            w_inh_plastic = clip(w_inh_plastic + eta*G_pre*z_istdp_pre, w_inh_plastic_min, w_inh_plastic_max)
            
            '''

        self.synapse_istdp_pre='''
            w_inh_plastic = clip(w_inh_plastic + eta*G_pre*(z_istdp_post+1), w_inh_plastic_min, w_inh_plastic_max)
            
            '''
    # def default(self, ):
    #     '''
    #     '''
        self.neuron_e = self.neuron

        self.neuron_i = _add_eq(self.neuron, self.neuron_global_rate_shared)

        self.synapse_e = _add_eq(self.synapse_ampa, self.synapse_triplet_stdp, )

        self.synapse_e_static = _add_eq(self.synapse_ampa,)

        self.synapse_i = _add_eq(self.synapse_gaba, self.synapse_istdp)

        self.synapse_e_pre = _add_eq(self.synapse_ampa_pre, self.synapse_triplet_stdp_pre)

        self.synapse_e_pre_static = _add_eq(self.synapse_ampa_pre_static,)

        self.synapse_e_post = _add_eq(self.synapse_triplet_stdp_post)

        self.synapse_i_pre = _add_eq(self.synapse_gaba_pre, self.synapse_istdp_pre)

        self.synapse_i_post = _add_eq(self.synapse_istdp_post)
    def ____init__(self, **kwargs):
        '''
        '''

        self.neuron ='''
            # subthreshold membrane voltage
            #-------------------------------
            du/dt = (E_L-u)/t_m + g_exc*(E_exc-u)/t_m + (g_gaba + g_a + g_b)*(E_gaba-u)/t_m : volt

            # threshold adaptation
            #------------------------
            du_thresh/dt = (u_thresh_rest-u_thresh)/t_u_thresh : volt

            # adaptation currents
            #----------------------
            dg_a/dt = -g_a/t_a :1
            dg_b/dt = -g_b/t_b :1

            # excitatory synaptic conductance
            #--------------------------------
            dg_ampa/dt = -g_ampa/t_ampa : 1
            dg_nmda/dt = (g_ampa-g_nmda)/t_nmda :1
            g_exc = alpha*g_ampa + (1-alpha)*g_nmda :1

            # inhibitory synaptic conductance
            #--------------------------------
            dg_gaba/dt = -g_gaba/t_gaba : 1

            # neuron specific plasticity traces
            #-----------------------------------
            # dz_plus/dt = -z_plus/t_plus : 1 
            # dz_minus/dt = -z_minus/t_minus : 1 
            # dz_slow/dt = -z_slow/t_slow : 1 
            # dz_istdp/dt = -z_istdp/t_istdp : 1 

            '''

        self.neuron_reset='''
            u_thresh=u_thresh_reset
            u = E_L
            g_a += Delta_a # adaptation 
            g_b += Delta_b # adaptation
            
            
            '''

        self.neuron_global_rate='''
            dH/dt=-H/t_H : 1 
            G = H-gamma:1 
            # gamma= 0:1
        '''
        self.synapse_global_rate='''
            
        '''
        self.synapse_global_rate_pre='''
            H += 1
        '''

        self.synapse_stp='''
            dx_stp/dt = (1-x_stp)/t_d  : 1 (event-driven)
            du_stp/dt = (U_stp-u_stp)/t_f : 1 (event-driven)
            
            '''  
        self.synapse_stp_pre='''
            x_stp += -u_stp*x_stp
            u_stp += U_stp*(1-u_stp)
            '''

        self.synapse_ampa = '''
            w_ampa :1
            '''

        self.synapse_gaba = '''
            w_gaba:1
            '''

        self.synapse_ampa_pre='''
            g_ampa += x_stp*u_stp*w_exc_plastic
            '''

        self.synapse_ampa_pre_static='''
            g_ampa += w_exc_plastic_init
            '''
        
        self.synapse_gaba_pre='''
            g_gaba += w_inh_plastic
            '''

        self.synapse_triplet_stdp='''
            w_exc_plastic :1 
            dz_plus/dt = -z_plus/t_plus : 1 (event-driven)
            dz_minus/dt = -z_minus/t_minus : 1 (event-driven)
            dz_slow/dt = -z_slow/t_slow : 1 (event-driven)
            dw_cons/dt = -w_cons/t_w_cons + w_exc_plastic/t_w_cons - P*w_cons*(w_P/2 - w_cons)*(w_P-w_cons)/t_w_cons : 1 (clock-driven)
            '''

        # self.synapse_triplet_stdp_post='''
        #     w_exc_plastic += A*z_plus*z_slow - Beta*(w_exc_plastic-w_consolidation)*z_minus**3
        #     '''

        self.synapse_triplet_stdp_post='''
            w_exc_plastic = clip(w_exc_plastic + A*z_plus*z_slow - Beta*(w_exc_plastic-w_cons)*z_minus**3, w_exc_plastic_min, w_exc_plastic_max)
            z_plus += 1
            z_slow += 1
            '''
        # self.synapse_triplet_stdp_pre='''
        #     w_exc_plastic += -(B*z_minus-delta)
        #     '''
        self.synapse_triplet_stdp_pre='''
            w_exc_plastic = clip(w_exc_plastic -(B*z_minus-delta), w_exc_plastic_min, w_exc_plastic_max)
            z_minus += 1
            '''

        self.synapse_istdp='''
            w_inh_plastic :1
            dzpre_istdp/dt = -zpre_istdp/t_istdp : 1 (event-driven)
            dzpost_istdp/dt = -zpost_istdp/t_istdp : 1 (event-driven)
            '''

        self.synapse_istdp_post='''
            w_exc_plastic = clip(w_inh_plastic + eta*G_pre*zpre_istdp, w_inh_plastic_min, w_inh_plastic_max)
            zpost_istdp += 1
            '''

        self.synapse_istdp_pre='''
            w_exc_plastic = clip(w_inh_plastic + eta*G_pre*(zpost_istdp+1), w_inh_plastic_min, w_inh_plastic_max)
            zpre_istdp += 1
            '''

        self.neuron_e = self.neuron

        self.neuron_i = _add_eq(self.neuron, self.neuron_global_rate)

        self.synapse_e = _add_eq(self.synapse_ampa, self.synapse_stp, self.synapse_triplet_stdp)

        self.synapse_e_static = _add_eq(self.synapse_ampa,)

        self.synapse_i = _add_eq(self.synapse_gaba, self.synapse_istdp)

        self.synapse_e_pre = _add_eq(self.synapse_ampa_pre, self.synapse_stp_pre, self.synapse_triplet_stdp_pre)

        self.synapse_e_pre_static = _add_eq(self.synapse_ampa_pre_static,)

        self.synapse_e_post = _add_eq(self.synapse_triplet_stdp_post)

        self.synapse_i_pre = _add_eq(self.synapse_gaba_pre, self.synapse_istdp_pre)

        self.synapse_i_post = _add_eq(self.synapse_istdp_post)

class Pokorny2019:
    '''
    '''
    def __init__(self, **kwargs):
        '''
        '''

        self.neuron ='''
            # firing rate
            #----------------
            rate = r_0*exp(r_slope*u):1
            # membrane potential
            #-----------------
            u = epsp - ipsp + E_exc + E_exc_generic : 1
            # epsp
            #---------
            epsp = epsp_r-epsp_f : 1
            depsp_r/dt =  -epsp_r/t_r : 1
            depsp_f/dt =  -epsp_f/t_f : 1
            # ipsp
            #------
            ipsp = ipsp_r-ipsp_f : 1
            dipsp_r/dt =  -ipsp_r/t_r : 1
            dipsp_f/dt =  -ipsp_f/t_f : 1

            tp = (t_f*t_r)/(t_r - t_f) * log(t_r/t_f)
            factor = -exp(-tp/t_f) + exp(-tp/t_r)
            factor = 1/factor

        '''

def _add_eq(*list_args):

        '''
        '''
        equation  = ''
        for eq in list_args:
            equation = equation + '\n' + eq 

        return equation 