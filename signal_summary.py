import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

def session_labeler(path):
    DMS = np.load(path+'/alf/_ibl_trials.DMS.npy')
    DLS = np.load(path+'/alf/_ibl_trials.DLS.npy')
    NAcc = np.load(path+'/alf/_ibl_trials.NAcc.npy')
    raw = pd.read_csv(path + '/alf/fp_data/FP470_processed.csv')
    raw = raw.loc[raw['include']==1]
    _, ax = plt.subplots(3,4)
    regions=[DMS,DLS,NAcc]
    regions_name = ['DMS','DLS','NAcc']
    max_y = np.nanmax(np.concatenate([DMS, DLS,NAcc]))
    min_y = np.nanmin(np.concatenate([DMS, DLS,NAcc]))
    frames = int(10/raw['Timestamp'].diff().median()) # Always 10 seconds
    for i, area in enumerate(regions):
        len_ses = len(area)
        len_factor = int(len_ses/4)
        plt.sca(ax[i,0])
        plt.plot(raw[regions_name[i]])
        plt.ylabel('Raw Fluorescence')
        plt.sca(ax[i,1])
        plt.plot(area)
        plt.ylabel('DF/F')
        plt.ylim(min_y, max_y)
        plt.title(regions_name[i])
        plt.sca(ax[i,2])
        plt.plot(area[len_factor-frames:len_factor])
        plt.ylim(min_y, max_y)
        plt.sca(ax[i,3])
        plt.plot(area[(len_factor*2)-frames:(len_factor*2)])
        plt.ylim(min_y, max_y)
    plt.tight_layout()
    plt.savefig(path+'/signal_summary.png')

if __name__ == "__main__":
    path = sys.argv[1]
    session_labeler(path)