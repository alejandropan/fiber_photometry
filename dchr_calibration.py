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
from scipy.stats import sem
import seaborn as sns

# Chrmine Calibration
ses = '//Volumes/witten/Alex/Data/calibration_chrmine/dchr2_opto_bandit_cs/calibration/003'
framerate  = 100
power_list = [1,0.75,0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.005]

# Functions
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'full') / w

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

def extract_everything (ses):
    # Load and rename FP files
    loc_dict={'Region2G': 'DMS','Region1G': 'DLS','Region0G': 'NAcc'}
    fp_data = pd.read_csv(ses + '/raw_fp_data/FP470')
    fp_data = fp_data.rename(columns=loc_dict)
    bpod_feedback_time = np.load(ses + '/alf/_ibl_trials.feedback_times.npy')
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

    # Patch session if needed: Delete short pulses (sample smaller than frame aquisition rate) or pulses before acquistion for FP and big breaks (acquistion started twice)
    signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1
    sample_ITI  = np.median(np.diff(signal.loc[signal['TTL_change']==1].index))
    # Update TTL change column
    signal['TTL_change'] = 0
    signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1
    assert abs(len(np.where(signal['DAQ_FP'].diff()==1)[0]) - len(fp_data)) < 12

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
    selection = run_all_regions(fp_data)
    selection = selection[2]
    if selection: #check if list is empty if not include everything
        fp_data['include'] = 0
    else:
        fp_data['include'] = 1
    for select in selection:
        fp_data.iloc[int(select[0]):int(select[1]), -1] = 1
    fp_data = fp_data.loc[fp_data['include']==1]
    fp_data['DMS_p'] =  bleach_correct(fp_data['DMS'], avg_window = 180, fr = sample_ITI)
    fp_data['NAcc_p'] =  bleach_correct(fp_data['NAcc'], avg_window = 180, fr = sample_ITI)
    fp_data['DLS_p'] =  bleach_correct(fp_data['DLS'], avg_window = 180, fr = sample_ITI)

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

    # Get opto timestamps
    start_of_bpod = np.where(signal['DAQ_bpod']>0)[0][0]
    opto =pd.DataFrame()
    opto['DA_pulse'] = np.nan
    opto['DA_train_start'] = np.nan
    opto['DA_power'] = np.nan
    opto['DAQ_opto'] = tdms_file._channel_data["/'Analog'/'AI7'"].data
    opto['DAQ_opto'] = 1 * (opto['DAQ_opto']>=2)

    opto_ups = np.where(opto['DAQ_opto'].diff()==1)[0]
    opto['DA_pulse'].iloc[opto_ups] = 1 
    opto_ups[np.where(np.diff(opto_ups)>51)]
    opto['DA_train_start'].iloc[opto_ups[19]] = 1 
    opto['DA_power'].iloc[opto_ups[19]] = power_list[0]

    assert len(power_list) == (len(np.where(np.diff(opto_ups)>51)[0])+1)/3

    power_repeats = []
    for j in np.arange(len(power_list)*3):
        power_repeats.append(power_list[int(j/3)])


    for i, pw in enumerate(power_repeats[1:]): # 1: because I already added the first in line 170
        ts  = opto_ups[np.where(np.diff(opto_ups)>51)][i]
        opto['DA_power'].iloc[ts] = pw
        opto['DA_train_start'].iloc[ts] = (i+1)%3 + 1

    print(opto.loc[~np.isnan(opto['DA_train_start'])])
    opto.loc[~np.isnan(opto['DA_train_start'])].reset_index().to_csv(ses+'/alf/opto_cal_sync.csv')

def plot_calibration(ses,ses_water, power_list):
    fig, ax = plt.subplots(2,3)
    # MOUSE 1
    power_list = [0.75,0.5,0.3, 0.1, 0.005]
    ses = '/Volumes/witten/Alex/Data/calibration_chrmine/dchr1_opto_bandit_cs/calibration/002'
    ses_water = '/Volumes/witten/Alex/Data/calibration_chrmine/dchr1_opto_bandit_cs/session/001'
    b,a = butter(4, 5/(100/2),'lowpass') # butter filter: order 4, 2hz objective frequency, lowpass
    cmap = plt.get_cmap('summer')
    opto_sync = pd.read_csv(ses+'/alf/opto_cal_sync.csv')
    dms = np.load(ses+'/alf/_ibl_trials.DMS.npy')
    dls = np.load(ses+'/alf/_ibl_trials.DLS.npy')
    nacc = np.load(ses+'/alf/_ibl_trials.NAcc.npy')
    water_timestamps = np.load(ses_water+'/alf/_ibl_trials.feedback_daq_times.npy')[np.load(ses_water+'/alf/_ibl_trials.feedbackType.npy')==1]
    water_nacc =  np.load(ses_water+'/alf/_ibl_trials.NAcc.npy')
    water_dms =  np.load(ses_water+'/alf/_ibl_trials.DMS.npy')
    water_dls =  np.load(ses_water+'/alf/_ibl_trials.DLS.npy')
    water_fp_timestamps =  np.load(ses_water+'/alf/_ibl_trials.DAQ_timestamps.npy')
    labels = ['NAcc','DMS','DLS']
    timestamps =  np.load(ses+'/alf/_ibl_trials.DAQ_timestamps.npy')
    fluos = [nacc,dms,dls]
    fluos_water = [water_nacc,water_dms,water_dls]
    for axes in np.arange(3):
        plt.sca(ax[0,axes])
        fluo = fluos[axes]
        fluo[np.where(np.isnan(fluo))[0]] = 0 #deal with nan
        fluo = filtfilt(b,a,fluo)
        for pw in power_list:
            idx = opto_sync.loc[opto_sync['DA_power']==pw , 'index'].to_numpy()
            idx = idx-1000 # the timstamps are for the final pulse
            reg = np.zeros([3,350])
            for i, ids in enumerate(idx):
                frame = np.where(abs(timestamps-ids)==np.min(abs(timestamps-ids)))[0][0]#closest frame
                reg[i,:] = fluo[frame-50:frame+300]
            m = np.mean(reg,axis=0)
            time = np.arange(-500,3000,10)
            error = sem(reg,axis=0)
            plt.plot(time, m, color=cmap(pw))
            plt.fill_between(time, m-error, m+error,  color=cmap(pw))
            plt.title(labels[axes])
            sns.despine()
        idx = water_timestamps[:100]
        reg = np.zeros([len(water_timestamps),175])
        fluo_water = fluos_water[axes]
        fluo_water[np.where(np.isnan(fluo_water))[0]] = 0 #deal with nan
        fluo_water = filtfilt(b,a,fluo_water)        
        for i, ids in enumerate(idx):
            frame = np.where(abs(water_fp_timestamps-ids)==np.min(abs(water_fp_timestamps-ids)))[0][0]#closest frame
            reg[i,:] = fluo_water[frame-25:frame+150]
        m = np.mean(reg,axis=0)
        time = np.arange(-500,3000,20)
        error = sem(reg,axis=0)
        plt.plot(time, m, color='k')
        plt.fill_between(time, m-error, m+error, color='k')
        plt.title(labels[axes])
        plt.xlabel('Time from outcome')
        plt.ylabel('DF/F')
        sns.despine()
    
    # MOUSE 2
    ses = '/Volumes/witten/Alex/Data/calibration_chrmine/dchr2_opto_bandit_cs/calibration/003'
    ses_water = '/Volumes/witten/Alex/Data/calibration_chrmine/dchr2_opto_bandit_cs/session/001'
    b,a = butter(4, 5/(100/2),'lowpass') # butter filter: order 4, 2hz objective frequency, lowpass
    opto_sync = pd.read_csv(ses+'/alf/opto_cal_sync.csv')
    dms = np.load(ses+'/alf/_ibl_trials.DMS.npy')
    dls = np.load(ses+'/alf/_ibl_trials.DLS.npy')
    nacc = np.load(ses+'/alf/_ibl_trials.NAcc.npy')
    water_timestamps = np.load(ses_water+'/alf/_ibl_trials.feedback_daq_times.npy')[:-1][np.load(ses_water+'/alf/_ibl_trials.feedbackType.npy')==1]
    water_nacc =  np.load(ses_water+'/alf/_ibl_trials.NAcc.npy')
    water_dms =  np.load(ses_water+'/alf/_ibl_trials.DMS.npy')
    water_dls =  np.load(ses_water+'/alf/_ibl_trials.DLS.npy')
    water_fp_timestamps =  np.load(ses_water+'/alf/_ibl_trials.DAQ_timestamps.npy')
    labels = ['NAcc','DMS','DLS']
    timestamps =  np.load(ses+'/alf/_ibl_trials.DAQ_timestamps.npy')
    fluos = [nacc,dms,dls]
    fluos_water = [water_nacc,water_dms,water_dls]
    for axes in np.arange(3):
        plt.sca(ax[1,axes])
        fluo = fluos[axes]
        fluo[np.where(np.isnan(fluo))[0]] = 0 #deal with nan
        fluo = filtfilt(b,a,fluo)
        for pw in power_list:
            idx = opto_sync.loc[opto_sync['DA_power']==pw , 'index'].to_numpy()
            idx = idx-1000 # the timstamps are for the final pulse
            reg = np.zeros([3,350])
            for i, ids in enumerate(idx):
                frame = np.where(abs(timestamps-ids)==np.min(abs(timestamps-ids)))[0][0]#closest frame
                reg[i,:] = fluo[frame-50:frame+300]
            m = np.mean(reg,axis=0)
            time = np.arange(-500,3000,10)
            error = sem(reg,axis=0)
            plt.plot(time, m,  color=cmap(pw))
            plt.fill_between(time, m-error, m+error,  color=cmap(pw))
            plt.title(labels[axes])
            sns.despine()
        idx = water_timestamps[:100]
        reg = np.zeros([len(water_timestamps),350])
        fluo_water = fluos_water[axes]
        fluo_water[np.where(np.isnan(fluo_water))[0]] = 0 #deal with nan
        fluo_water = filtfilt(b,a,fluo_water)        
        for i, ids in enumerate(idx):
            frame = np.where(abs(water_fp_timestamps-ids)==np.min(abs(water_fp_timestamps-ids)))[0][0]#closest frame
            reg[i,:] = fluo_water[frame-50:frame+300]
        m = np.mean(reg,axis=0)
        time = np.arange(-500,3000,10)
        error = sem(reg,axis=0)
        plt.plot(time, m, color='k')
        plt.fill_between(time, m-error, m+error, color='k')
        plt.title(labels[axes])
        plt.xlabel('Time from outcome')
        plt.ylabel('DF/F')
        sns.despine()
    
