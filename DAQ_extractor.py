## Objectives of optimization:
# Measure difference between LED times and DAQ times to begin with
# Check for skipped frames

from nptdms import TdmsFile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ibllib.io.extractors.biased_trials import extract_all
from ibllib.io.extractors.training_trials import (
    Choice, FeedbackTimes, FeedbackType, GoCueTimes, GoCueTriggerTimes,
    IncludedTrials, Intervals, ItiDuration, ProbabilityLeft, ResponseTimes, RewardVolume,
    StimOnTimes_deprecated, StimOnTriggerTimes, StimOnOffFreezeTimes, ItiInTimes,
    StimOffTriggerTimes, StimFreezeTriggerTimes, ErrorCueTriggerTimes)
from ibllib.io.extractors import camera
from ibllib.io import ffmpeg
from ibllib.io.extractors.training_wheel import extract_all as extract_all_wheel
from pathlib import Path
from distutils.dir_util import copy_tree
import os
import sys
from scipy.signal import butter, filtfilt
from  session_crop import run_all_regions
import json
from signal_summary import session_labeler

# Functions

def dff_qc (dff, thres=0.01, thres_z=3, frame_interval=20):
    "Check if there is a transient 3x s.d or higher every 10 min if not exclude"
    peaks = np.where(dff>thres)[0]
    peaks_std = np.where((dff/np.nanstd(dff))>thres_z)[0]
    ten_min_window = 10*60*1000/frame_interval
    bins = np.arange(0,len(dff),ten_min_window)
    qc = 1
    for i in np.arange(1,len(bins)):
        if len(np.where(np.logical_and(peaks>=bins[i-1], peaks<=bins[i]))[0])<1:
            qc = 0
    for i in np.arange(1,len(bins)):
        if len(np.where(np.logical_and(peaks_std>=bins[i-1], peaks_std<=bins[i]))[0])<1:
            qc = 0
    return qc

def bleach_correct(nacc, avg_window = 60, fr = 25):
    '''
    Correct for bleaching of gcamp across the session. Calculates
    DF/F
    Parameters
    ----------
    nacc: series with fluorescence data
    avg_window: time for sliding window to calculate F value in seconds
    fr = frame_rate
    '''
    # First calculate sliding window
    avg_window = int(avg_window*fr)
    F = nacc.rolling(avg_window, center = True).mean()
    nacc_corrected = (nacc - F)/F
    return nacc_corrected


def extract_fp_daq(ses, save=True, correct_bleaching=True, framerate=50):
    '''
    Extract FP data, aligned to bpod and to DAQ
    ses: path to session data (not fp folder) (str)
    save: whether to save new alf files
    '''
    # Load and rename FP files
    loc_dict={'Region2G': 'DLS','Region1G': 'NAcc','Region0G': 'DMS'}
    try:
        fp_data = pd.read_csv(ses + '/raw_fp_data/FP470')
    except:
        fp_data = pd.read_csv(ses + '/raw_fp_data/FP470.csv')
    fp_data = fp_data.rename(columns=loc_dict)
    bpod_feedback_time = np.load(ses + '/alf/_ibl_trials.feedback_times.npy')

    # Save location dictionary for signal summary
    with open(ses +'/alf/loc_dict.json', 'w') as fp:
        json.dump(loc_dict, fp)

    inter_frames  = np.median(np.diff(fp_data['FrameCounter']))

    if (len(np.unique(np.diff(fp_data['FrameCounter'])))>1):
        print('WARNING: %d 470 frames skipped' \
            %len(np.where(np.diff(fp_data['FrameCounter'])!= inter_frames)[0]))
    # Load DAQ file
    for file in os.listdir(ses + '/raw_fp_data/'):
        if file.endswith(".tdms"):
            td_f = file
    tdms_file = TdmsFile.read(ses + '/raw_fp_data/'+ td_f)
    signal =pd.DataFrame()
    signal['DAQ_FP'] = tdms_file._channel_data["/'Analog'/'AI0'"].data
    signal['DAQ_bpod'] = tdms_file._channel_data["/'Analog'/'AI1'"].data
    signal['DAQ_FP'] = 1 * (signal['DAQ_FP']>=4)
    signal['DAQ_bpod'] = 1 * (signal['DAQ_bpod']>=2)

    ### Patch session if needed: Delete short pulses (sample smaller than frame aquisition rate) or pulses before acquistion for FP and big breaks (acquistion started twice)
    signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1
    sample_ITI  = np.median(np.diff(signal.loc[signal['TTL_change']==1].index))

    if (sample_ITI==10): #New protocol saves ITI for all: 470,145 and 2x empty frames
        true_FP = signal.loc[signal['TTL_change']==1].index[::int((1000/framerate)/sample_ITI)]
        signal['TTL_change']= 0
        signal['DAQ_FP']= 0
        signal.iloc[true_FP,signal.columns.get_loc('TTL_change')]=1
        signal.iloc[true_FP,signal.columns.get_loc('DAQ_FP')]=1
        signal.iloc[true_FP+1,signal.columns.get_loc('DAQ_FP')]=1 # Pulses are 2ms long
        signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1
        sample_ITI  = np.median(np.diff(signal.loc[signal['TTL_change']==1].index))
    print(sample_ITI)

    while np.diff(signal.loc[signal['TTL_change']==1].index).max()>sample_ITI*10: #Session was started twice
        print('Session started twice')
        ttl_id = np.where(np.diff(signal.loc[signal['TTL_change']==1].index) ==
                 np.diff(signal.loc[signal['TTL_change']==1].index).max())[0][0]
        real_id = signal.loc[signal['TTL_change']==1].index[ttl_id]
        signal.iloc[:int(real_id+np.diff(signal.loc[signal['TTL_change']==1].index).max()
                         -sample_ITI),:] = 0

    ## Check for skipped frames
    max_inter_led = np.diff(signal.loc[signal['TTL_change']==1].index).max()
    min_inter_led = np.diff(signal.loc[signal['TTL_change']==1].index).min()
    max_inter_frame = np.diff(fp_data.Timestamp*1000).max()
    min_inter_frame = np.diff(fp_data.Timestamp*1000).min()
    if ((max_inter_led - min_inter_led)/sample_ITI >= 1)|((max_inter_frame - min_inter_frame)/sample_ITI >= 1):
        if (max_inter_led - min_inter_led)/sample_ITI >= 1:
            print(ses + ' skipped LEDs')
        if (max_inter_led - min_inter_led)/sample_ITI >= 1:
            print(ses+' skipped LEDs')
        led_len = len(signal.loc[signal['TTL_change']==1].index)
        skipped_leds = np.where(np.diff(signal.loc[signal['TTL_change']==1].index[:(led_len-50)])>=(sample_ITI*2))[0] #[:-50] is to ignore the end led extra pulses
        skipped_frames = np.where(np.diff(fp_data.Timestamp[:(led_len-50)]*1000)>=(sample_ITI*2))[0]
        if np.equal(skipped_leds, skipped_frames)==False:
            return print('ERROR: skipped frames dont match TTLs')
        else:
            print('skipped frames match ttls, uff')            

    ##
    pulse_to_del = \
        signal.loc[signal['TTL_change']==1].index[np.where((np.diff(signal.loc[signal['TTL_change']==1].index)<sample_ITI*0.95) |
             (np.diff(signal.loc[signal['TTL_change']==1].index)>sample_ITI*1.05))[0]]

    for i in pulse_to_del:
        signal.iloc[i:int(i+sample_ITI*1.05), np.where(signal.columns=='DAQ_FP')[0]]=0

    # Update TTL change column
    signal['TTL_change'] = 0
    signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1

    led_number = len(pd.read_csv(ses + '/raw_fp_data/output_0_timestamps', header=None)[::2])


    assert abs(len(np.where(signal['DAQ_FP'].diff()==1)[0]) - len(fp_data)) < 15

    # Align events
    fp_data['DAQ_timestamp'] = np.nan
    daq_idx = fp_data.columns.get_loc('DAQ_timestamp')
    fp_data.iloc[:,daq_idx] = \
        np.where(signal['DAQ_FP'].diff()==1)[0][:len(fp_data)]

    # Extract Trial Events
    signal.loc[np.where(signal['DAQ_bpod'].diff()==1)[0], 'bpod_on'] = 1
    signal.loc[np.where(signal['DAQ_bpod'].diff()==-1)[0], 'bpod_off'] = 1
    signal.loc[np.where(signal['DAQ_bpod'].diff()==1)[0], 'bpod_duration'] = \
        signal.loc[signal['bpod_off']==1].index - \
        signal.loc[signal['bpod_on']==1].index
    signal['feedbackTimes'] = np.nan
    signal.loc[signal['bpod_duration']>105, 'feedbackTimes'] = 1
    signal['bpod_event'] = np.nan
    signal.loc[signal['bpod_duration']>1000, 'bpod_event'] = 'error'
    signal.loc[signal['bpod_duration']<=105, 'bpod_event'] = 'trial_start'
    signal.loc[(signal['bpod_duration']>100) &
               (signal['bpod_duration']<1000), 'bpod_event'] = 'reward'

    # Interpolate times from bpod clock
    assert abs((len(signal.loc[signal['feedbackTimes']==1]) - \
        len(bpod_feedback_time))) <=1
    signal['bpod_time'] = np.nan
    choices = np.load(ses+'/alf/_ibl_trials.choice.npy') 
    nan_trials  = np.where(choices==0)[0] # No choice was made
    if len(nan_trials) != 0:
            try: # For new code with bpod pulses also in NO GOs
                signal.loc[signal['feedbackTimes']==1, 'bpod_time'] = bpod_feedback_time
            except: # For older code without pulses in nan
                if len(bpod_feedback_time)> len(signal.loc[signal['feedbackTimes']==1]):
                    bpod_feedback_time = np.delete(bpod_feedback_time, nan_trials)

    if (len(signal.loc[signal['feedbackTimes']==1]) - \
        len(bpod_feedback_time)) == 0:
        signal.loc[signal['feedbackTimes']==1, 'bpod_time'] = bpod_feedback_time
    if (len(signal.loc[signal['feedbackTimes']==1]) - \
              len(bpod_feedback_time)) == 1: # If bpod didn't save last trial
        print('Bpod missing last trial')
        signal.loc[signal['feedbackTimes']==1, 'bpod_time'] = np.append(bpod_feedback_time,np.nan)
    signal['bpod_time'].interpolate(inplace=True)
    # Delete values after last known bpod - interpolate will not extrapolate!
    signal.iloc[np.where(signal['bpod_time'] == signal['bpod_time'].max())[0][1]:,
                signal.columns.get_loc('bpod_time')] = np.nan
    # Align fluo data with bpod time
    fp_data['bpod_time'] = np.nan
    daq_idx = fp_data.columns.get_loc('bpod_time')
    fp_data.iloc[:,daq_idx] = \
        signal['bpod_time'].to_numpy()[np.where(signal['DAQ_FP'].diff()==1)[0][:len(fp_data)]]

    # Correct for bleaching
    if correct_bleaching==True:
        #cropping
        selection = run_all_regions(fp_data)
        selection = selection[2]
        if selection: #check if list is empty if not include everything
            fp_data['include'] = 0
        else:
            fp_data['include'] = 1
        for select in selection:
            fp_data.iloc[int(select[0]):int(select[1]), -1] = 1
        fp_data = fp_data.loc[fp_data['include']==1]
        fp_data['DMS_p'] =  bleach_correct(fp_data['DMS'], avg_window = 60, fr = sample_ITI)
        fp_data['NAcc_p'] =  bleach_correct(fp_data['NAcc'], avg_window = 60, fr = sample_ITI)
        fp_data['DLS_p'] =  bleach_correct(fp_data['DLS'], avg_window = 60, fr = sample_ITI)

    #comparison
    #plt.plot(bleach_correct(fp_data['NAcc'].to_numpy(), fp_data['NAcc_isos'].to_numpy()))
    #plt.plot(bleach_correct_old(fp_data['NAcc'], avg_window = 120, fr = sample_ITI)+1)

    qc = {'DMS':dff_qc(fp_data['DMS_p'], frame_interval=sample_ITI), 
    'NAcc':dff_qc(fp_data['NAcc_p'], frame_interval=sample_ITI), 
    'DLS':dff_qc(fp_data['DLS_p'], frame_interval=sample_ITI)}
    with open(ses +'/alf/FP_QC.json', 'w') as fp:
        json.dump(qc, fp)
    
    # Save
    if save == True:
        np.save(ses+'/alf/_ibl_trials.feedback_daq_times.npy',
                signal.loc[signal['feedbackTimes']==1].index.to_numpy())
        np.save(ses+'/alf/_ibl_trials.goCueTrigger_daq_times.npy',
                signal.loc[signal['bpod_event']=='trial_start'].index.to_numpy())
        np.save(ses+'/alf/_ibl_trials.error_daq_times.npy',
                signal.loc[signal['bpod_event']=='error'].index.to_numpy())
        np.save(ses+'/alf/_ibl_trials.reward_daq_times.npy',
                signal.loc[signal['bpod_event']=='reward'].index.to_numpy())
        np.save(ses+'/alf/_ibl_trials.DLS.npy',
                fp_data['DLS_p'].to_numpy())
        np.save(ses+'/alf/_ibl_trials.DMS.npy',
                fp_data['DMS_p'].to_numpy())
        np.save(ses+'/alf/_ibl_trials.NAcc.npy',
                fp_data['NAcc_p'].to_numpy())
        np.save(ses+'/alf/_ibl_trials.DAQ_timestamps.npy',
                fp_data['DAQ_timestamp'].to_numpy())
        np.save(ses+'/alf/_ibl_fluo.times.npy',
                fp_data['bpod_time'].to_numpy())
        fp_data.to_csv(ses+'/raw_fp_data/FP470_processed.csv')
