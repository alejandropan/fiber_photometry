import pandas as pd
import sys
import os
from pathlib import Path
from os import listdir
from os.path import isfile, join
import json

#from nptdms import TdmsFile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from distutils.dir_util import copy_tree
import os
import sys
import json

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

def fix_f0(ses):
    fp_data = pd.read_csv(ses+'/alf/fp_data/FP470_processed.csv')
    fr = int(1/fp_data['Timestamp'].diff().mean())
    fp_data['DMS_p'] =  bleach_correct(fp_data['DMS'], avg_window = 60, fr = fr)
    fp_data['NAcc_p'] =  bleach_correct(fp_data['NAcc'], avg_window = 60, fr = fr)
    fp_data['DLS_p'] =  bleach_correct(fp_data['DLS'], avg_window = 60, fr = fr)
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
    fp_data.to_csv(ses+'/alf/fp_data/FP470_processed.csv')
