from nptdms import TdmsFile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ibllib.io.extractors.biased_trials import extract_all
from ibllib.io.extractors.training_wheel import extract_all as extract_all_wheel
from pathlib import Path
from distutils.dir_util import copy_tree
import os

# Functions

def bleach_correct(nacc, avg_window = 120, fr = 30):
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
    F = nacc.rolling(12000, center = True).mean()
    nacc_corrected = (nacc - F)/F
    return nacc_corrected

def move_fp_data_to_alf(ses, dry=True):
    # Moves FP data from FP folder to alf folder for that animal
    alf_path = Path(ses)
    date = alf_path.parent.name
    mouse = alf_path.parent.parent.name
    root = alf_path.parent.parent.parent.parent.parent
    fp_path = root.joinpath('Photometry_Data', 'session_data',
                            mouse, date, '001')
    assert fp_path.exists()==True
    if dry==False:
        if Path(ses+'/alf/fp_data').exists()==False:
            os.mkdir(ses+'/alf/fp_data')
        destination= ses+'/alf/fp_data'
        copy_tree(str(fp_path),destination)

def extract_fp_daq(ses, save=False, correct_bleaching=True):
    '''
    Extract FP data, aligned to bpod and to DAQ
    ses: path to session data (not fp folder) (str)
    save: whether to save new alf files
    '''
    # Load and rename FP files
    try:
        fp_data = pd.read_csv(ses + '/alf/fp_data/FP470')
    except:
        fp_data = pd.read_csv(ses + '/alf/fp_data/FP470.csv')
    fp_data = fp_data.rename(columns={'Region0G': 'DMS',
                              'Region1G': 'NAcc',
                              'Region2G': 'DLS'})
    try:
        fp_data_415 = pd.read_csv(ses + '/alf/fp_data/FP415')
    except:
        fp_data_415 = pd.read_csv(ses + '/alf/fp_data/FP415.csv')
    fp_data_415 = fp_data_415.rename(columns={'Region0G': 'DMS_isos',
                                      'Region1G': 'NAcc_isos',
                                      'Region2G': 'DLS_isos'})
    bpod_feedback_time = np.load(ses + '/alf/_ibl_trials.feedback_times.npy')

    assert (len(fp_data) - len(fp_data_415)) < 2 # Check that intermitent channels were not skipped
    fp_data[['DMS_isos','NAcc_isos','DLS_isos']] = np.nan
    fp_data.iloc[:len(fp_data_415),-3:] = \
        fp_data_415[['DMS_isos','NAcc_isos','DLS_isos']]

    # Load DAQ file
    for file in os.listdir(ses + '/alf/fp_data/'):
        if file.endswith(".tdms"):
            td_f = file
    tdms_file = TdmsFile.read(ses + '/alf/fp_data/'+ td_f)
    signal =pd.DataFrame()
    signal['DAQ_FP'] = tdms_file._channel_data["/'Analog'/'AI0'"].data
    signal['DAQ_bpod'] = tdms_file._channel_data["/'Analog'/'AI1'"].data
    signal['DAQ_FP'] = 1 * (signal['DAQ_FP']>=4)
    signal['DAQ_bpod'] = 1 * (signal['DAQ_bpod']>=2)

    ### Patch session if needed: Delete short pulses (sample smaller than frame aquisition rate) or pulses before acquistion for FP and big breaks (acquistion started twice)
    signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1
    sample_ITI  = np.median(np.diff(signal.loc[signal['TTL_change']==1].index))

    while np.diff(signal.loc[signal['TTL_change']==1].index).max()>sample_ITI*4: #Session was started twice
        print('Session started twice')
        ttl_id = np.where(np.diff(signal.loc[signal['TTL_change']==1].index) ==
                 np.diff(signal.loc[signal['TTL_change']==1].index).max())[0][0]
        real_id = signal.loc[signal['TTL_change']==1].index[ttl_id]
        signal.iloc[:int(real_id+np.diff(signal.loc[signal['TTL_change']==1].index).max()
                         -sample_ITI),:] = 0

    pulse_to_del = \
        signal.loc[signal['TTL_change']==1].index[np.where((np.diff(signal.loc[signal['TTL_change']==1].index)<sample_ITI*0.95) |
             (np.diff(signal.loc[signal['TTL_change']==1].index)>sample_ITI*1.05))[0]]
    for i in pulse_to_del:
        print(len(pulse_to_del) noise frames)
        signal.iloc[i:int(i+sample_ITI*1.05), np.where(signal.columns=='DAQ_FP')[0]]=0
    # Update TTL change column
    signal['TTL_change'] = 0
    signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1
    assert abs(len(np.where(signal['DAQ_FP'].diff()==1)[0]) - len(fp_data)) < 6

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
    signal.loc[signal['bpod_duration']>100, 'feedbackTimes'] = 1
    signal['bpod_event'] = np.nan
    signal.loc[signal['bpod_duration']>1000, 'bpod_event'] = 'error'
    signal.loc[signal['bpod_duration']<=100, 'bpod_event'] = 'trial_start'
    signal.loc[(signal['bpod_duration']>100) &
               (signal['bpod_duration']<1000), 'bpod_event'] = 'reward'

    # Interpolate times from bpod clock
    assert (len(signal.loc[signal['feedbackTimes']==1]) - \
        len(bpod_feedback_time)) <=1
    signal['bpod_time'] = np.nan
    choices = np.load(ses+'/alf/_ibl_trials.choice.npy')
    nan_trials  = np.where(choices==0)[0] # No choice was made
    if len(nan_trials) != 0:
            try: # For new code with bpod pulses also in NO GOs
                signal.loc[signal['feedbackTimes']==1, 'bpod_time'] = bpod_feedback_time
            except: # For older code without pulses in nan
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
        fp_data['DMS'] = bleach_correct(fp_data['DMS'], avg_window = 120, fr = sample_ITI)
        fp_data['NAcc'] = bleach_correct(fp_data['NAcc'], avg_window = 120, fr = sample_ITI)
        fp_data['DLS'] = bleach_correct(fp_data['DLS'], avg_window = 120, fr = sample_ITI)
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
                fp_data['DLS'].to_numpy())
        np.save(ses+'/alf/_ibl_trials.DMS.npy',
                fp_data['DMS'].to_numpy())
        np.save(ses+'/alf/_ibl_trials.NAcc.npy',
                fp_data['NAcc'].to_numpy())
        np.save(ses+'/alf/_ibl_trials.DAQ_timestamps.npy',
                fp_data['DAQ_timestamp'].to_numpy())
        np.save(ses+'/alf/_ibl_fluo.times.npy',
                fp_data['bpod_time'].to_numpy())


if __name__ == "__main__":
    ses = sys.argv[1]
    extract_all(ses, save=True)
    extract_all_wheel(ses, save=True)
    move_fp_data_to_alf(ses, dry=False)
    extract_fp_daq(ses, save=True)
