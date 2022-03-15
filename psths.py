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


def divide_in_trials(fluo_times, cue_times, nacc, t_before_epoch = 0.1):
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
    #nacc_f = stats.zscore(nacc_f,nan_policy='omit')

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
            t_range = [-0.1, 1.0], T_BIN = 0.04, plot = False):
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
        fluo = [fluo_in_trials[i][times == j].sum() for j in np.unique(times)]
        t[i,np.unique(times)] = fluo

    # Exclude first and last bin which include everything outside epoch
    t = t[:,1:-1]

    # Generate X axis
    x = np.arange(t_range[0], t_range[1], T_BIN)[:-1]

    # Plot
    if plot == True:
        plt.plot(x,np.nanmean(t,0))

    return t, x

def plot_psth(condition1, color=None, alpha=1, plot_error=True, linestyle=None):
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
    plt.ylabel('DF/F')

### Formal analysis ####

# Opto vs reward

def opto_vs_reward(opto_block,cue_times, feedback, feedback_times,fluo_time_in_trials, fluo_in_trials,
                 subtract_baseline =  True, save = True):

    opto = np.where(opto_block ==0)[0]
    reward = np.where(opto_block == 1)[0]
    correct = np.where(feedback == 1)[0]
    opto = np.intersect1d(opto, correct)[:100]
    reward = np.intersect1d(reward, correct)[:100]

    condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times, trial_list = opto)
    condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials,  cue_times, trial_list = reward)

    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])


    fig, ax = plt.subplots(1,2, figsize=(18,10), sharey=True)
    plt.sca(ax[0])
    ax[0].set_title('Go Cue', fontsize=25)
    plot_psth(condition1)
    plot_psth(condition2)

    condition1 = fp_psth(fluo_time_in_trials,
                         fluo_in_trials,
                         feedback_times, trial_list = opto)
    condition2 = fp_psth(fluo_time_in_trials,
                         fluo_in_trials,
                         feedback_times, trial_list = reward)



    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])

    plt.sca(ax[1])
    ax[1].set_title('Reward Cue', fontsize=25)
    plot_psth(condition1)
    plot_psth(condition2)
    if save == True:
        fig.savefig('optovsreward.pdf')





# psth easy vs hard correct trials  stim and feedback

def easy_vs_hard(signed_contrast,cue_times, feedback, feedback_times,fluo_time_in_trials, fluo_in_trials,
                 subtract_baseline =  True, save = True):
    '''
    Figure function plots Psth for stim and feedback from easy
    and hard trials
    '''
    easy = np.where(signed_contrast >= 0.25)[0]
    hard = np.where(signed_contrast < 0.25 )[0]
    correct = np.where(feedback == 1)[0]
    easy = np.intersect1d(easy, correct)
    hard = np.intersect1d(hard, correct)

    condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times, trial_list = easy)
    condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials,  cue_times, trial_list = hard)

    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])


    fig, ax = plt.subplots(1,2, figsize=(18,10), sharey=True)
    plt.sca(ax[0])
    ax[0].set_title('Go Cue', fontsize=25)
    plot_psth(condition1)
    plot_psth(condition2)

    condition1 = fp_psth(fluo_time_in_trials,
                         fluo_in_trials,
                         feedback_times, trial_list = easy)
    condition2 = fp_psth(fluo_time_in_trials,
                         fluo_in_trials,
                         feedback_times, trial_list = hard)

    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])

    plt.sca(ax[1])
    ax[1].set_title('Reward Cue', fontsize=25)
    plot_psth(condition1)
    plot_psth(condition2)
    if save == True:
        fig.savefig('easyvshard.pdf')



def ipsi_vs_contra(signed_contrast,cue_times, feedback,fluo_time_in_trials, fluo_in_trials,
                 subtract_baseline =  True, save = True):
    '''
    Figure function plots Psth for stim and feedback from easy
    and hard trials
    '''

    easy_r = np.where(signed_contrast == 1)[0]
    easy_l = np.where(signed_contrast == -1)[0]
    condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                         trial_list = easy_r)
    condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                         trial_list = easy_l)

    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1),
                        (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1),
                        (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])

    fig, ax = plt.subplots(1,2, figsize=(15,10))
    plt.sca(ax[0])
    ax[0].set_title('Go Cue',fontsize=25)
    plot_psth(condition1, color = 'k')
    plot_psth(condition2)

    easy_r = np.where(signed_contrast == 1)[0]
    easy_l = np.where(signed_contrast == -1)[0]
    condition1 = fp_psth(fluo_time_in_trials,fluo_in_trials, feedback_times,
                         trial_list = easy_r)
    condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials, feedback_times,
                         trial_list = easy_l)

    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1),
                        (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1),
                        (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])

    plt.sca(ax[1])
    ax[1].set_title('Reward Cue',fontsize=25)
    plot_psth(condition1, color = 'k')
    plot_psth(condition2)
    plt.title('blue is right DMS')
    if save == True:
        fig.savefig('ipsivscontra.pdf')


def ratio_stim_feedback(signed_contrast, feedback, fluo_in_trials,
                        fluo_time_in_trials):

    easy = np.where(signed_contrast >= 0.5)[0]
    hard = np.where(signed_contrast <= 0.125)[0]
    correct = np.where(feedback == 1)[0]
    easy = np.intersect1d(easy, correct)
    hard = np.intersect1d(hard, correct)
    condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                         cue_times, trial_list = easy)
    condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                         cue_times, trial_list = hard)
    F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1),
                    (len(condition1[0]), 1))
    F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1),
                    (len(condition2[0]), 1))
    condition1 = (condition1[0]-F1, condition1[1])
    condition2 = (condition2[0]-F2, condition2[1])

    f_condition1 = fp_psth(fluo_time_in_trials,fluo_in_trials,
                           feedback_times, trial_list = easy)
    f_condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                           feedback_times, trial_list = hard)
    F1 = np.reshape(np.nanmean(f_condition1[0][:,0:3],1),
                    (len(f_condition1[0]), 1))
    F2 = np.reshape(np.nanmean(f_condition2[0][:,0:3],1),
                    (len(f_condition2[0]), 1))
    f_condition1 = (condition1[0]-F1, condition1[1])
    f_condition2 = (condition2[0]-F2, condition2[1])

    easy_stim = np.nanmax(condition1[0][:,3:12],1)
    easy_stim = easy_stim[~np.isnan(easy_stim)]
    easy_feedback = np.nanmax(f_condition1[0][:,3:12],1)
    easy_feedback = easy_feedback[~np.isnan(easy_feedback)]
    hard_stim = np.nanmax(condition2[0][:,3:12],1)
    hard_stim = hard_stim[~np.isnan(hard_stim)]
    hard_feedback = np.nanmax(f_condition2[0][:,3:12],1)
    hard_feedback = hard_feedback[~np.isnan(hard_feedback)]

    fig, ax = plt.subplots(1,2, figsize=(15,10))
    plt.sca(ax[0])
    ax[0].set_title('Easy',fontsize=25)
    bars_easy = [np.mean(easy_stim), np.mean(easy_feedback)]
    err_easy = [stats.sem(easy_stim), stats.sem(easy_feedback)]
    plt.bar(['stim', 'feedback'], bars_easy, yerr=err_easy)
    plt.ylabel('DF/F z-score',fontsize=25)

    plt.sca(ax[1])
    ax[1].set_title('Hard',fontsize=25)
    bars_hard = [np.mean(hard_stim), np.mean(hard_feedback)]
    err_hard = [stats.sem(hard_stim), stats.sem(hard_feedback)]
    plt.bar(['stim', 'feedback'], bars_hard, yerr=err_hard)
    plt.ylabel('DF/F z-score',fontsize=25)


# Cumulative performance

def ongoing_performance(feedback):
    cum_perf = np.empty(len(feedback))
    cum_perf[:] = np.nan
    for i in range(len(feedback)):
        if np.isnan(feedback[i]):
            continue
        else:
            if i == 0:
                cum_perf[i] = feedback[i]
            else:
                cum_perf[i] = feedback[i] + cum_perf[i-1]
    perf = cum_perf/(np.arange(len(cum_perf))+1)

    return perf

# Performance for that contrast and feedback gcamp for correct
def contrast_expectation_gcamp_response(signed_contrast,feedback,
                               fluo_time_in_trials,
                               fluo_in_trials,
                               feedback_times):
    '''
    feedback times is just wathever epoch you want, it should be called 'epoch'
    '''
    easy_raw = np.where(signed_contrast >= 0.5)[0]
    correct = np.where(feedback == 1)[0]
    easy = np.intersect1d(easy_raw, correct)
    f_condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                           feedback_times, trial_list = easy)
    F1 = np.reshape(np.nanmean(f_condition1[0][:,0:3],1),
                    (len(f_condition1[0]), 1))
    f_condition1 = (f_condition1[0]-F1, f_condition1[1])
    easy_stim = np.nanmax(f_condition1[0][:,3:12],1)
    easy_stim = easy_stim[~np.isnan(easy_stim)]
    perf = ongoing_performance(np.searchsorted(easy_raw, easy))

    sig = stats.pearsonr(perf,easy_stim)

    fig, ax = plt.subplots(1, figsize=(15,10))
    plt.sca(ax)
    ax.set_title('Gcamp_expectation_for_contrast',fontsize=25)
    plt.scatter(perf,easy_stim, c ='k')
    plt.plot(np.unique(perf), np.poly1d(np.polyfit(perf, easy_stim, 1))(np.unique(perf)))
    plt.xlabel('Ongoing performance', fontsize=25)
    plt.ylabel('Reward cue DF/F',fontsize=25)
    plt.legend(sig)

    return sig

# overall Performance fand feedback gcamp
def expectation_gcamp_response(signed_contrast,feedback,
                               fluo_time_in_trials, fluo_in_trials,feedback_times):
    easy_raw = np.where(signed_contrast >= 0.5)[0]
    correct = np.where(feedback == 1)[0]
    easy = np.intersect1d(easy_raw, correct)
    f_condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                           feedback_times, trial_list = easy)
    F1 = np.reshape(np.nanmean(f_condition1[0][:,0:3],1),
                    (len(f_condition1[0]), 1))
    f_condition1 = (f_condition1[0]-F1, f_condition1[1])
    easy_stim = np.nanmax(f_condition1[0][:,3:12],1)
    easy_stim = easy_stim[~np.isnan(easy_stim)]
    perf = ongoing_performance(feedback)
    plt.scatter(perf[easy],easy_stim)
    sig  = stats.pearsonr(perf[easy],easy_stim)
    return sig


def after_error_correct(signed_contrast, cue_times, feedback, feedback_times,fluo_time_in_trials, fluo_in_trials,
                 subtract_baseline =  True, save = True):
    '''
    Figure function plots Psth for stim and feedback from easy
    and hard trials
    '''

    noevidence = np.where(signed_contrast == 0)[0]
    correct = np.where(feedback == 1)[0]
    incorrect = np.where(feedback == -1)[0]
    after_correct = correct + 1
    after_incorrect = incorrect + 1
    noevidence_after_correct = np.intersect1d(noevidence, after_correct)
    noevidence_after_incorrect = np.intersect1d(noevidence, after_incorrect)


    condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                           cue_times, trial_list = after_correct[:-1], plot = False)

    condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                           cue_times, trial_list =  after_incorrect[:-1], plot= False)



    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])


    fig, ax = plt.subplots(1,2, figsize=(20,10))
    plt.sca(ax[0])
    ax[0].set_title('Go Cue all trials', fontsize=25)
    plot_psth(condition1, color='k')
    plot_psth(condition2)

    condition1 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                           cue_times, trial_list = noevidence_after_correct[:-1], plot = False)

    condition2 = fp_psth(fluo_time_in_trials, fluo_in_trials,
                           cue_times, trial_list =  noevidence_after_incorrect[:-1], plot= False)


    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])

    plt.sca(ax[1])
    ax[1].set_title('Go Cue 0% contrast trials', fontsize=25)
    plot_psth(condition1, color='k')
    plot_psth(condition2)

    if save == True:
        fig.savefig('after_error_correct_0.pdf')




def by_contrast(signed_contrast,cue_times,fluo_time_in_trials, fluo_in_trials):
    '''
    Figure function plots Psth for stim and feedback from easy
    and hard trials
    There is baseline subtractions always
    '''

    for contrast in np.unique(signed_contrast):
        c = np.where(signed_contrast == contrast)[0]
        condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, cue_times,
                         trial_list = c)
        F1 = np.reshape(np.nanmean(condition_psth[0][:,0:3],1),
                        (len(condition_psth[0]), 1))
        condition_psth = (condition_psth[0]-F1, condition_psth[1])
        plt.title('Go Cue',fontsize=25)
        if contrast>0:
            plot_psth(condition_psth, color = 'r', alpha=abs(contrast))
        else:
            plot_psth(condition_psth, color = 'dodgerblue', alpha=abs(contrast))

        sns.despine()


if __name__=='Main':
    # Input some data
    session = ['/Volumes/witten/Alex/Data/Subjects/fip_14/2021-03-22/001']

    for i, ses in enumerate(session):
            fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
            nacc = np.load(ses +'/alf/_ibl_trials.DLS.npy')
            cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
            response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
            feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
            left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
            right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
            l_trials = np.nan_to_num(left_trials)
            r_trials = np.nan_to_num(right_trials)
            signed_contrast = r_trials - l_trials
            feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
            fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, nacc, t_before_epoch = 0.1)
            cue_times = cue_times[:-1]
            response_times = response_times[:-1]
            feedback_times = feedback_times[:-1]
            l_trials = l_trials[:-1]
            r_trials = r_trials[:-1]
            signed_contrast = signed_contrast[:-1]
            feedback = feedback[:-1]
            stimOn_times  = np.load(ses + '/alf/_ibl_trials.stimOn_times.npy')

    for contrast in np.unique(signed_contrast):
            c = np.where(signed_contrast == contrast)[0]
            d = np.where(feedback ==1 )[0]
            c = np.intersect1d(c,d)
            condition_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, feedback_times,
                                trial_list = c)
            F1 = np.reshape(np.nanmean(condition_psth[0][:,0:3],1),
                                (len(condition_psth[0]), 1))
            condition_psth = (condition_psth[0]-F1, condition_psth[1])
            plt.title('Go Cue',fontsize=25)
            if contrast>0:
                plot_psth(condition_psth, color = 'r', alpha=abs(contrast))
            else:
                plot_psth(condition_psth, color = 'dodgerblue', alpha=abs(contrast))
            sns.despine()


    # Note:  Remember to cut down last trial from above after running divide in trials
