import numpy as np
import pandas as pd
import os
from nptdms import TdmsFile

import sys
from pathlib import Path
from os import listdir
from os.path import isfile, join

def skipped_frames_checker(ses, framerate=50):
    '''
    Extract FP data, aligned to bpod and to DAQ
    ses: path to session data (not fp folder) (str)
    save: whether to save new alf files
    '''
    # Load and rename FP files
    loc_dict={'Region2G': 'DMS','Region1G': 'DLS','Region0G': 'NAcc'}
    try:
        fp_data = pd.read_csv(ses + '/raw_fp_data/FP470')
    except:
        fp_data = pd.read_csv(ses + '/raw_fp_data/FP470.csv')
    fp_data = fp_data.rename(columns=loc_dict)


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
        print('problem')
        if (max_inter_led - min_inter_led)/sample_ITI >= 1:
            print(ses + ' skipped LEDs')
        if (max_inter_led - min_inter_led)/sample_ITI >= 1:
            print(ses+' skipped LEDs')
    else:
        print(ses + ' all good!')


topdir = sys.argv[1]
os.chdir(topdir)
all_subdirs = [x[0] for x in os.walk('./')]

for sdx, sessiondir in enumerate(all_subdirs):
    if 'raw_video_data' not in sessiondir: # Skip all directories that don't end in raw_video_data
        continue
    print(sessiondir[:-14])
    try:
        skipped_frames_checker(sessiondir[:-14])
    except:
        print('files_missing')