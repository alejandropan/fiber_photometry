#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:21:16 2020

@author: alex
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

fluo_times = np.load('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_fluo.times.npy')
nacc = np.load('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_loc2.fluo.npy')
cue_times = np.load ('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_trials.goCueTrigger_times.npy')
response_times = np.load('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_trials.response_times.npy')
feedback_times = np.load('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_trials.feedback_times.npy')
left_trials = np.load('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_trials.contrastLeft.npy')
right_trials = np.load('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_trials.contrastRight.npy')
l_trials = np.nan_to_num(left_trials)
r_trials = np.nan_to_num(right_trials)
signed_contrast = r_trials - l_trials
feedback = np.load('/Users/alex/Documents/Trial_dataset_fip1_FP/005/alf/_ibl_trials.feedbackType.npy')

def bleach_correct(nacc, avg_window = 60):
    '''
    Correct for bleaching of gcamp across the session. Calculates
    DF/F
    Parameters
    ----------
    nacc: alf file with fluorescence values for the area of interest
    avg_window: time for sliding window to calculate F value
    '''
    # First calculate sliding window
    F = np.convolve(nacc, np.ones((avg_window,))/avg_window, mode='same')
    nacc_corrected = nacc/F
    return nacc_corrected
        
        
def divide_in_trials(fluo_times, cue_times, nacc):
    '''
    Makes list of list with fluorescnece and timestamps divided by trials
    
    Parameters
    ----------
    
    Fluo_times: alf file with times for fluorescence frames
    cue_times:  alf file with times for cue (the start of the trial)
    nacc: alf file with fluorescence values for the area of interest
    '''
    #Delete before first trial and last trial

    fluo_times_s = np.delete(fluo_times,np.where(fluo_times < cue_times[0]))
    nacc_s = np.delete(nacc,np.where(fluo_times < cue_times[0]))
    fluo_times_f = np.delete(fluo_times_s,np.where(fluo_times_s > cue_times[-1]))
    nacc_f = np.delete(nacc_s,np.where(fluo_times_s > cue_times[-1]))
    nacc_f = stats.zscore(nacc_f)
    
    fluo_in_trials = []
    fluo_time_in_trials = []
    for i in range(len(cue_times)): 
        # Get from 0.3 before
        if i == len(cue_times) -1 :
            fluo_in_trials.append(nacc_f[(fluo_times_f 
                                     >= cue_times[i] - 0.3)])
            fluo_time_in_trials.append(fluo_times_f[(fluo_times_f 
                                     >= cue_times[i] - 0.3)])
        
        else:
            fluo_in_trials.append(nacc_f[(fluo_times_f 
                                         >= cue_times[i] - 0.3) & 
                   (fluo_times_f <= cue_times[i+1])])
            fluo_time_in_trials.append(fluo_times_f[(fluo_times_f 
                                         >= cue_times[i] - 0.3) & 
                   (fluo_times_f <= cue_times[i+1])])
    return fluo_in_trials, fluo_time_in_trials



def fp_psth(fluo_time_in_trials, cue_times, trial_list = np.arange(len(cue_times)),
            t_range = [-0.1, 1.0], T_BIN = 0.03, plot = False):
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
    T_BIN : Size of the bin. The default is 0.03s (frame rate)
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


def plot_psth(condition1):
    '''
    Generates psth with shaded error bar
    condition1 : np array with fluorescence across bins
    '''
    error = stats.sem(condition1[0], nan_policy='omit')
    y = np.nanmean(condition1[0], axis=0)
    x = condition1[1]
    plt.plot(x, y)
    plt.fill_between(x, y-error, y+error)
    plt.xlabel('time(s)')
    plt.ylabel('DF/F z-score')





### Formal analysis ####


# psth easy vs hard correct trials  stim and feedback

def easy_vs_hard(signed_contrast,feedback,fluo_time_in_trials,
                 subtract_baseline =  True):
    '''
    Figure function plots Psth for stim and feedback from easy 
    and hard trials
    '''
    easy = np.where(signed_contrast == 1)[0]
    hard = np.where(signed_contrast == 0.125)[0]
    correct = np.where(feedback == 1)[0]
    easy = np.intersect1d(easy, correct)
    hard = np.intersect1d(hard, correct)
    
    condition1 = fp_psth(fluo_time_in_trials, cue_times, trial_list = easy)
    condition2 = fp_psth(fluo_time_in_trials, cue_times, trial_list = hard)
    
    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])
    
    
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    plt.sca(ax[0])
    ax[0].set_title('Go Cue')
    plot_psth(condition1)
    plot_psth(condition2) 
    
    condition1 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = easy)    
    condition2 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = hard)   
    
    if subtract_baseline ==  True:
        F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
        F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
        condition1 = (condition1[0]-F1, condition1[1])
        condition2 = (condition2[0]-F2, condition2[1])
    
    plt.sca(ax[1])
    ax[1].set_title('Reward Cue')
    plot_psth(condition1)
    plot_psth(condition2) 


# ipsilateral vs contra easy trial stim
def easy_vs_hard(signed_contrast,feedback,fluo_time_in_trials,
                 subtract_baseline =  True):
    '''
    Figure function plots Psth for stim and feedback from easy 
    and hard trials
    '''
    easy = np.where(signed_contrast == 1)[0]
    hard = np.where(signed_contrast == 0.125)[0]
    correct = np.where(feedback == 1)[0]
    easy = np.intersect1d(easy, correct)
    hard = np.intersect1d(hard, correct)
    easy_r = np.where(signed_contrast == 1)[0]
    easy_l = np.where(signed_contrast == -1)[0]
    condition1 = fp_psth(fluo_time_in_trials, cue_times, trial_list = easy_r)
    condition2 = fp_psth(fluo_time_in_trials, cue_times, trial_list = easy_l)
    
    F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
    F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
    
    condition1 = (condition1[0]-F1, condition1[1])
    condition2 = (condition2[0]-F2, condition2[1])
    
    fig, ax = plt.subplots()
    plot_psth(condition1)
    plot_psth(condition2) 

# ipsilateral vs contra easy trial feedback
easy_r = np.where(signed_contrast == 1)[0]
easy_l = np.where(signed_contrast == -1)[0]
condition1 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = easy_r)    
condition2 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = easy_l)    



F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))


condition1 = (condition1[0]-F1, condition1[1])
condition2 = (condition2[0]-F2, condition2[1])


fig, ax = plt.subplots()
plot_psth(condition1)
plot_psth(condition2) 


# correct vs incorrect easy and ipsi+contra, cue
easy = np.where(abs(signed_contrast) >= 0.5)[0] 
correct = np.where(feedback == 1)[0] 
incorrect = np.where(feedback == -1)[0] 
e_correct = np.intersect1d(easy, correct)
e_incorrect = np.intersect1d(easy, incorrect)
condition1 = fp_psth(fluo_time_in_trials, cue_times, trial_list = e_correct)
condition2 = fp_psth(fluo_time_in_trials, cue_times, trial_list = e_incorrect)


F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))


condition1 = (condition1[0]-F1, condition1[1])
condition2 = (condition2[0]-F2, condition2[1])


fig, ax = plt.subplots()
plot_psth(condition1)
plot_psth(condition2) 

# correct vs incorrect easy and ipsi+contra, feedback
easy = np.where(abs(signed_contrast) >= 0.5)[0] 
correct = np.where(feedback == 1)[0] 
incorrect = np.where(feedback == -1)[0] 
e_correct = np.intersect1d(easy, correct)
e_incorrect = np.intersect1d(easy, incorrect)
condition1 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = e_correct)
condition2 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = e_incorrect)


F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))


condition1 = (condition1[0]-F1, condition1[1])
condition2 = (condition2[0]-F2, condition2[1])


fig, ax = plt.subplots()
plot_psth(condition1)
plot_psth(condition2) 


# Ratio Fedback stim
easy = np.where(signed_contrast >= 0.5)[0]
hard = np.where(signed_contrast <= 0.125)[0]
correct = np.where(feedback == 1)[0]
easy = np.intersect1d(easy, correct)
hard = np.intersect1d(hard, correct)
condition1 = fp_psth(fluo_time_in_trials, cue_times, trial_list = easy)
condition2 = fp_psth(fluo_time_in_trials, cue_times, trial_list = hard)
F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
condition1 = (condition1[0]-F1, condition1[1])
condition2 = (condition2[0]-F2, condition2[1])


f_condition1 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = easy)
f_condition2 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = hard)
F1 = np.reshape(np.nanmean(f_condition1[0][:,0:3],1), (len(f_condition1[0]), 1))
F2 = np.reshape(np.nanmean(f_condition2[0][:,0:3],1), (len(f_condition2[0]), 1))
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

for i in range(len(easy_stim)):
    plt.plot([1,2],[easy_stim[i],easy_feedback[i]])
    
for i in range(len(hard_stim)):
    plt.plot([1,2],[hard_stim[i],hard_feedback[i]])


bars_easy = [np.mean(easy_stim), np.mean(easy_feedback)]
err_easy = [stats.sem(easy_stim), stats.sem(easy_feedback)]
plt.bar(['stim', 'feedback'], bars_easy, yerr=err_easy)

bars_hard = [np.mean(hard_stim), np.mean(hard_feedback)]
err_hard = [stats.sem(hard_stim), stats.sem(hard_feedback)]
plt.bar(['stim', 'feedback'], bars_hard, yerr=err_hard)

## Performance in bins with ratio


easy = np.where(signed_contrast >= 0.5)[0]
hard = np.where(signed_contrast <= 0.125)[0]
condition1 = fp_psth(fluo_time_in_trials, cue_times, trial_list = easy)
condition2 = fp_psth(fluo_time_in_trials, cue_times, trial_list = hard)
F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
condition1 = (condition1[0]-F1, condition1[1])
condition2 = (condition2[0]-F2, condition2[1])


f_condition1 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = easy)
f_condition2 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = hard)
F1 = np.reshape(np.nanmean(f_condition1[0][:,0:3],1), (len(f_condition1[0]), 1))
F2 = np.reshape(np.nanmean(f_condition2[0][:,0:3],1), (len(f_condition2[0]), 1))
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
easy_raw = np.where(signed_contrast >= 0.5)[0]
correct = np.where(feedback == 1)[0]
easy = np.intersect1d(easy_raw, correct)
f_condition1 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = easy)
F1 = np.reshape(np.nanmean(f_condition1[0][:,0:3],1), (len(f_condition1[0]), 1))
f_condition1 = (f_condition1[0]-F1, f_condition1[1])
easy_stim = np.nanmax(f_condition1[0][:,3:12],1)
easy_stim = easy_stim[~np.isnan(easy_stim)]
perf = ongoing_performance(np.searchsorted(easy_raw, easy))
plt.scatter(perf,easy_stim)
stats.pearsonr(perf,easy_stim)


# overall Performance fand feedback gcamp
easy_raw = np.where(signed_contrast >= 0.5)[0]
correct = np.where(feedback == 1)[0]
easy = np.intersect1d(easy_raw, correct)
f_condition1 = fp_psth(fluo_time_in_trials, feedback_times, trial_list = easy)
F1 = np.reshape(np.nanmean(f_condition1[0][:,0:3],1), (len(f_condition1[0]), 1))
f_condition1 = (f_condition1[0]-F1, f_condition1[1])
easy_stim = np.nanmax(f_condition1[0][:,3:12],1)
easy_stim = easy_stim[~np.isnan(easy_stim)]
perf = ongoing_performance(feedback)
plt.scatter(perf[easy],easy_stim)
stats.pearsonr(perf[easy],easy_stim)


# overall Performance fand feedback gcamp for correct
for i in range(len(easy_stim)):
    plt.plot([1,2],[easy_stim[i],easy_feedback[i]])
    
for i in range(len(hard_stim)):
    plt.plot([1,2],[hard_stim[i],hard_feedback[i]])


bars_easy = [np.mean(easy_stim), np.mean(easy_feedback)]
err_easy = [stats.sem(easy_stim), stats.sem(easy_feedback)]
plt.bar(['stim', 'feedback'], bars_easy, yerr=err_easy)

bars_hard = [np.mean(hard_stim), np.mean(hard_feedback)]
err_hard = [stats.sem(hard_stim), stats.sem(hard_feedback)]
plt.bar(['stim', 'feedback'], bars_hard, yerr=err_hard)