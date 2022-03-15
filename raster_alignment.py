#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:51:03 2020

@author: alex
"""

import numpy as np 

fluorescence_times =  np.load('/Volumes/witten/Alex/Data/Subjects/_iblrig_test_mouse/2020-11-12/017/alf/_ibl_fluo.times.npy')

fluorescence_times =  np.load('/Volumes/witten/Alex/Data/Subjects/_iblrig_test_mouse/2020-11-12/017/alf/_ibl_leftCamera.times.npy')
fluorescence = np.load('/Volumes/witten/Alex/Data/Subjects/_iblrig_test_mouse/2020-11-12/017/alf/_ibl_loc2.fluo.npy')
goStim_times = np.load('/Volumes/witten/Alex/Data/Subjects/_iblrig_test_mouse/2020-11-12/017/alf/_ibl_trials.stimOn_times.npy')
visual_trigger = np.load('/Volumes/witten/Alex/Data/Subjects/_iblrig_test_mouse/2020-11-12/017/alf/_ibl_trials.stimOnTrigger_times.npy')
sound_trigger = np.load('/Volumes/witten/Alex/Data/Subjects/_iblrig_test_mouse/2020-11-12/017/alf/_ibl_trials.goCueTrigger_times.npy')
left = np.load('/Volumes/witten/Alex/Data/Subjects/_iblrig_test_mouse/2020-11-12/017/alf/_ibl_trials.contrastLeft.npy')

# Meter is on Right stim so I have to exclude left times
visual_trigger_right = visual_trigger[np.isnan(left)]


#divide fluorescence in trials
fluo_trials, fluo_time_trials = divide_in_trials(fluorescence_times, visual_trigger, 
                                        fluorescence, t_before_epoch = 0.3)

# Detect start from fluo
fp_gotime=np.empty(len(visual_trigger))
for i in range(len(visual_trigger[:-1])):
    dx = np.concatenate([np.array([np.nan]),np.diff(fluo_trials[i])])
    fp_gotime[i] = fluo_time_trials[i][np.where(abs(dx)==np.nanmax(abs(dx[:13])))[0][0]] #13 Limits for first 400ms
    

# Detect stim time from fluorescence
for j,i in enumerate(np.where(np.isnan(left))[0][1:]):
    plt.vlines(fp_gotime[i]-visual_trigger[i],j,j+1)
    plt.vlines(0,j,j+1, color='r')
    plt.xlim(-0.14,0.01)
    
    
i=4

plt.plot(fluo_time_trials[i], fluo_trials[i])
plt.vlines(fp_gotime[i],0,5, color='r')
plt.vlines(visual_trigger[i],0,5, color='b')
plt.vlines(sound_trigger[i],0,5, color='k')
plt.xlim(30, 31)


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

    fluo_times_s = np.delete(fluo_times,np.where(fluo_times < (cue_times[0] - t_before_epoch)))
    nacc_s = np.delete(nacc,np.where(fluo_times < (cue_times[0] - t_before_epoch)))
    fluo_times_f = np.delete(fluo_times_s,np.where(fluo_times_s > cue_times[-1]))
    nacc_f = np.delete(nacc_s,np.where(fluo_times_s > cue_times[-1]))
    nacc_f = stats.zscore(nacc_f)
    
    fluo_in_trials = []
    fluo_time_in_trials = []
    for i in range(len(cue_times)-1): 
        # Get from 0.3 before
            fluo_in_trials.append(nacc_f[(fluo_times_f 
                                         >= (cue_times[i] - t_before_epoch)) & 
                   (fluo_times_f < cue_times[i+1])])
            fluo_time_in_trials.append(fluo_times_f[(fluo_times_f 
                                         >= (cue_times[i] - t_before_epoch)) & 
                   (fluo_times_f < cue_times[i+1])])
    return fluo_in_trials, fluo_time_in_trials