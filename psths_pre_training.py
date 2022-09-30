#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:21:16 2020

@author: alex
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import sklearn as sk
import seaborn as sns
from scipy import  stats

def divide_in_trials(fluo_times, cue_times, nacc, t_before_epoch = 0.5, zscore=True):
    '''
    Makes list of list with fluorescnece and timestamps divided by trials
    Deletes last trial since it might be incomplte
    Parameters
    ----------

    Fluo_times: alf file with times for fluorescence frames
    cue_times:  alf file with times for cue (the start of the trial)
    nacc: alf file with fluorescence values for the area of interest
    '''
    #Delete before first trial and last trial
    to_del = ~np.isnan(fluo_times)
    fluo_times = fluo_times[to_del]
    nacc = nacc[to_del]
    fluo_times_s = np.delete(fluo_times,np.where(fluo_times < cue_times[0]))
    nacc_s = np.delete(nacc,np.where(fluo_times < cue_times[0]))
    fluo_times_f = np.delete(fluo_times_s,np.where(fluo_times_s > cue_times[-1]))
    nacc_f = np.delete(nacc_s,np.where(fluo_times_s > cue_times[-1]))
    if zscore==True:
        nacc_f = stats.zscore(nacc_f,nan_policy='omit')
    fluo_in_trials = []
    fluo_time_in_trials = []
    for i in range(len(cue_times)-1):
        # Get from 0.3 before
            fluo_in_trials.append(nacc_f[(fluo_times_f
                                         >= cue_times[i] - t_before_epoch) &
                   (fluo_times_f <= cue_times[i+1])])
            fluo_time_in_trials.append(fluo_times_f[(fluo_times_f
                                         >= cue_times[i] - t_before_epoch) &
                   (fluo_times_f <= cue_times[i+1])])
    return fluo_in_trials, fluo_time_in_trials

def fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times, trial_list=None,
            t_range = [-0.5, 1.0], T_BIN = 0.04, plot = False):
    '''

    Parameters
    ----------
    fluo_time_in_trials : List of lists,
    size trials, each has the times of the frames assigned
    to that trial
    cue_times: array  size n=trials with the event time
    trial_list : trials to include (optional).
    The default is np.arange(len(cue_times)).
    t_range : Time range for fp_psth. The default is [-0.1, 1.0].
    T_BIN : Size of the bin. The default is 0.04s (frame rate)
    plot: Whether to plot a preliminary PSTH at the end

    Returns
    -------
    Plots psth
    t: array size rows= trials, columns = bins
    x: x-axis with bin times

    '''

    # Binned array with fluorescence values
    bins_len = round((t_range[1] - t_range[0])/T_BIN)
    bins = np.arange(t_range[0], t_range[1], T_BIN)
    cue_fluo_time = fluo_time_in_trials - cue_times
    t = np.empty([len(cue_times), bins_len+1])
    t[:] = np.nan
    for i in trial_list:
        times = np.digitize(cue_fluo_time[i], bins)
        fluo = [fluo_in_trials[i][times == j].mean() for j in np.unique(times)]
        t[i,np.unique(times)] = fluo

    # Exclude first and last bin which include everything outside epoch
    t = t[:,1:-1]

    # Generate X axis
    x = np.arange(t_range[0], t_range[1], T_BIN)[:-1]

    # Plot
    if plot == True:
        plt.plot(x,np.nanmean(t,0))

    return t, x

def plot_psth(condition1, color=None, alpha=1, plot_error=True, linestyle=None, zscore=True):
    '''
    Generates psth with shaded error bar
    condition1 : np array with fluorescence across bins
    '''
    y = np.nanmean(condition1[0], axis=0)
    x = condition1[1]
    plt.plot(x, y, color=color, alpha=alpha, linestyle=linestyle)
    if plot_error==True:
        error = stats.sem(condition1[0], nan_policy='omit')
        plt.fill_between(x, y-error, y+error, color=color, alpha=alpha)
    plt.xlabel('time(s)')
    if zscore==True:
        plt.ylabel('z-score DF/F')
    else:
        plt.ylabel('DF/F')

def plot_stim_psth(sessions, hems, zscore=True):
    fig, ax = plt.subplots(2,len(sessions),sharex=True, sharey=True)
    for i, ses in enumerate(sessions):
        ipsi = hems[i]
        fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
        fluo_data = np.load(ses +'/alf/_ibl_trials.DMS.npy')
        cue_times = np.load (ses +'/alf/_ibl_trials.goCuetrigger_times.npy')
        response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
        feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
        left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
        right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
        l_trials = np.nan_to_num(left_trials)
        r_trials = np.nan_to_num(right_trials)
        signed_contrast = r_trials - l_trials
        feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, fluo_data, t_before_epoch = 0.5, zscore=zscore)
        cue_times = cue_times[:-1]
        response_times = response_times[:-1]
        feedback_times = feedback_times[:-1]
        l_trials = l_trials[:-1]
        r_trials = r_trials[:-1]
        signed_contrast = signed_contrast[:-1]
        feedback = feedback[:-1]
        # Make signed contrast into (- ipsi), (+ contra)
        if ipsi=='left':
            signed_contrast=signed_contrast
        else:
            signed_contrast=signed_contrast*-1
        col_ipsi = ["#94D2BD", "#0A9396","#005F73", "#001219"]
        col_contra = ["#F48C06", "#DC2F02", "#9D0208", "#370617"]
        plt.sca(ax[0,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == contrast)[0]
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Contra'+ ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_contra[j], zscore=zscore)
            sns.despine()
        plt.sca(ax[1,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == -contrast)[0]
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Ipsi' + ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_ipsi[j], zscore=zscore)
            sns.despine()
        plt.tight_layout(w_pad=0.9)

    fig, ax = plt.subplots(2,len(sessions),sharex=True)
    for i, ses in enumerate(sessions):
        ipsi = hems[i]
        fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
        fluo_data = np.load(ses +'/alf/_ibl_trials.DLS.npy')
        cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
        response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
        feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
        left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
        right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
        l_trials = np.nan_to_num(left_trials)
        r_trials = np.nan_to_num(right_trials)
        signed_contrast = r_trials - l_trials
        feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, fluo_data, t_before_epoch = 0.1, zscore=zscore)
        cue_times = cue_times[:-1]
        response_times = response_times[:-1]
        feedback_times = feedback_times[:-1]
        l_trials = l_trials[:-1]
        r_trials = r_trials[:-1]
        signed_contrast = signed_contrast[:-1]
        feedback = feedback[:-1]
        # Make signed contrast into (- ipsi), (+ contra)
        if ipsi=='right':
            signed_contrast=signed_contrast
        else:
            signed_contrast=signed_contrast*-1
        col_ipsi = ["#94D2BD", "#0A9396","#005F73", "#001219"]
        col_contra = ["#F48C06", "#DC2F02", "#9D0208", "#370617"]
        plt.sca(ax[0,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == contrast)[0]
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Contra'+ ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_contra[j], zscore=zscore)
            sns.despine()
        plt.sca(ax[1,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == -contrast)[0]
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Ipsi' + ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_ipsi[j], zscore=zscore)
            sns.despine()
        plt.tight_layout(w_pad=0.9)


    fig, ax = plt.subplots(2,len(sessions),sharex=True)
    for i, ses in enumerate(sessions):
        ipsi = hems[i]
        fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
        fluo_data = np.load(ses +'/alf/_ibl_trials.NAcc.npy')
        cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
        response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
        feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
        left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
        right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
        l_trials = np.nan_to_num(left_trials)
        r_trials = np.nan_to_num(right_trials)
        signed_contrast = r_trials - l_trials
        feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, fluo_data, t_before_epoch = 0.1, zscore=zscore)
        cue_times = cue_times[:-1]
        response_times = response_times[:-1]
        feedback_times = feedback_times[:-1]
        l_trials = l_trials[:-1]
        r_trials = r_trials[:-1]
        signed_contrast = signed_contrast[:-1]
        feedback = feedback[:-1]
        # Make signed contrast into (- ipsi), (+ contra)
        if ipsi=='right':
            signed_contrast=signed_contrast
        else:
            signed_contrast=signed_contrast*-1
        col_ipsi = ["#94D2BD", "#0A9396","#005F73", "#001219"]
        col_contra = ["#F48C06", "#DC2F02", "#9D0208", "#370617"]
        plt.sca(ax[0,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == contrast)[0]
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Contra'+ ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_contra[j], zscore=zscore)
            sns.despine()
        plt.sca(ax[1,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == -contrast)[0]
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Ipsi' + ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_ipsi[j], zscore=zscore)
            sns.despine()
        plt.tight_layout(w_pad=0.9)

   
   

def plot_stim_pst_rewarded_unrewarded(sessions, hems, zscore=True):
    fig, ax = plt.subplots(2,len(sessions),sharex=True, sharey=True)
    for i, ses in enumerate(sessions):
        ipsi = hems[i]
        fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
        fluo_data = np.load(ses +'/alf/_ibl_trials.DMS.npy')
        cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
        response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
        feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
        left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
        right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
        l_trials = np.nan_to_num(left_trials)
        r_trials = np.nan_to_num(right_trials)
        signed_contrast = r_trials - l_trials
        feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, fluo_data, t_before_epoch = 0.5, zscore=zscore)
        cue_times = cue_times[:-1]
        response_times = response_times[:-1]
        feedback_times = feedback_times[:-1]
        l_trials = l_trials[:-1]
        r_trials = r_trials[:-1]
        signed_contrast = signed_contrast[:-1]
        feedback = feedback[:-1]
        # Make signed contrast into (- ipsi), (+ contra)
        if ipsi=='left':
            signed_contrast=signed_contrast
        else:
            signed_contrast=signed_contrast*-1
        col_ipsi = ["#94D2BD", "#0A9396","#005F73", "#001219"]
        col_contra = ["#F48C06", "#DC2F02", "#9D0208", "#370617"]
        plt.sca(ax[0,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == contrast)[0]
            f = np.where(feedback == 1)[0]
            c = np.intersect1d(c,f)
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Contra'+ ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_contra[j], zscore=zscore)
            sns.despine()
        plt.sca(ax[1,i])
        for j, contrast in enumerate(np.unique(abs(signed_contrast))):
            c = np.where(signed_contrast == contrast)[0]
            f = np.where(feedback == -1)[0]
            c = np.intersect1d(c,f)
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                                trial_list = c)
            plt.title('Ipsi' + ' ' +  ses[35:41])
            plot_psth(condition_psth, color = col_ipsi[j], zscore=zscore)
            sns.despine()
        plt.tight_layout(w_pad=0.9)


def plot_reward_psth_rewarded_unrewarded(sessions, hems, zscore=True):
    fig, ax = plt.subplots(2,len(sessions),sharex=True, sharey=True)
    for i, ses in enumerate(sessions):
        ipsi = hems[i]
        fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
        fluo_data = np.load(ses +'/alf/_ibl_trials.DLS.npy')
        cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
        response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
        feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
        left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
        right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
        l_trials = np.nan_to_num(left_trials)
        r_trials = np.nan_to_num(right_trials)
        signed_contrast = r_trials - l_trials
        feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, fluo_data, t_before_epoch = 0.5, zscore=zscore)
        cue_times = cue_times[:-1]
        response_times = response_times[:-1]
        feedback_times = feedback_times[:-1]
        l_trials = l_trials[:-1]
        r_trials = r_trials[:-1]
        signed_contrast = signed_contrast[:-1]
        feedback = feedback[:-1]
        # Make signed contrast into (- ipsi), (+ contra)
        if ipsi=='left':
            signed_contrast=signed_contrast
        else:
            signed_contrast=signed_contrast*-1
        col_ipsi = ["#94D2BD", "#0A9396","#005F73", "#001219"]
        col_contra = ["#F48C06", "#DC2F02", "#9D0208", "#370617"]
        plt.sca(ax[0,i])
        c = np.where(feedback == 1)[0]
        condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, feedback_times,
                            trial_list = c)
        plt.title('Rewarded'+ ' ' +  ses[35:41])
        plot_psth(condition_psth, color = col_contra[0], zscore=zscore)
        sns.despine()
        plt.sca(ax[1,i])
        c = np.where(feedback == -1)[0]
        condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, feedback_times,
                            trial_list = c)
        plt.title('Unrewarded' + ' ' +  ses[35:41])
        plot_psth(condition_psth, color = col_ipsi[0], zscore=zscore)
        sns.despine()
        plt.tight_layout(w_pad=0.9)

  # Note:  Remember to cut down last trial from above after running divide in trials



### Formal analysis ####

sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_26/pre_training/2022-03-29/without_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_27/pre_training/2022-03-29/without_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_28/pre_training/2022-03-30/without_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_29/pre-training/2022-03-29/without_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_30/wheel_fixed',
'/Volumes/witten/Alex/Data/Subjects/fip_32/pre_training/003',
'/Volumes/witten/Alex/Data/Subjects/fip_31/wheel_fixed',
'/Volumes/witten/Alex/Data/Subjects/fip_33/pre_training/fixed_wheel_shorter_ITI']

hems = ['left','left','left','left','right','right','right','right']
# Pre-training wheel fixed

plot_stim_psth(sessions, hems)

sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_26/pre_training/2022-03-29/with_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_27/pre_training/2022-03-29/with_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_28/pre_training/2022-03-30/with_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_29/pre-training/2022-03-29/with_valve_sound',
'/Volumes/witten/Alex/Data/Subjects/fip_30/wheel_moving',
'/Volumes/witten/Alex/Data/Subjects/fip_32/pre_training/004',
'/Volumes/witten/Alex/Data/Subjects/fip_31/wheel_moving',
'/Volumes/witten/Alex/Data/Subjects/fip_33/pre_training/fixed_wheel_shorter_ITI']
# Pre-training wheel moving

plot_stim_psth(sessions, hems)

sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_26/2022-04-03/001',
'/Volumes/witten/Alex/Data/Subjects/fip_27/2022-03-30/001',
'/Volumes/witten/Alex/Data/Subjects/fip_28/2022-03-31/001',
'/Volumes/witten/Alex/Data/Subjects/fip_29/2022-03-30/001',
'/Volumes/witten/Alex/Data/Subjects/fip_30/2022-05-19/003',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-05-19/001',
'/Volumes/witten/Alex/Data/Subjects/fip_31/2022-05-19/001',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-05-19/002']
# Fist session
plot_stim_psth(sessions, hems)


########
hems = ['right','right','right','left','left']

sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_20/2021-10-13/001',
'/Volumes/witten/Alex/Data/Subjects/fip_21/2021-10-15/001',
'/Volumes/witten/Alex/Data/Subjects/fip_22/2021-10-13/001',
'/Volumes/witten/Alex/Data/Subjects/fip_23/2021-12-08/005',
'/Volumes/witten/Alex/Data/Subjects/fip_25/2021-12-08/002'
]
# other animals 1sts sessions
plot_stim_psth(sessions, hems)

sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_20/pre-training/001',
'/Volumes/witten/Alex/Data/Subjects/fip_21/pre_training/2021-10-12/001',
'/Volumes/witten/Alex/Data/Subjects/fip_22/pre-training/2021-10-12/001',
'/Volumes/witten/Alex/Data/Subjects/fip_23/pre-training/2021-12-07/001',
'/Volumes/witten/Alex/Data/Subjects/fip_25/pre-training/2021-12-06/001']
# other animals pre-training
plot_stim_psth(sessions, hems)