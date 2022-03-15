import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from psths import *
from matplotlib.patches import Patch
import scipy.signal


DATA_FOLDER = '/Volumes/witten/Alex/Data/Subjects/'
MOUSE_FOLDER= '/Volumes/witten/Alex/Data/Subjects/fip_17'

def autocorr(x):
    x = x[~np.isnan(x)]
    result = np.correlate(x, x, mode='full')
    result = result/result.max()
    return result[int(result.size/2):]

def load_mouse_dataset(MOUSE_FOLDER):
    mouse_folder = Path(MOUSE_FOLDER)
    data = pd.DataFrame()
    list_subfolders_with_paths = [f.path for f in os.scandir(mouse_folder) if f.is_dir()] 
    list_subfolders_with_paths.sort()
    counter=0
    for fld in list_subfolders_with_paths:
        try:
            ses = pd.DataFrame()
            path = fld+'/001'
            DMS = np.load(path+'/alf/_ibl_trials.DMS.npy')
            DLS = np.load(path+'/alf/_ibl_trials.DLS.npy')
            NAcc = np.load(path+'/alf/_ibl_trials.NAcc.npy')
            fluo_times = np.load(path+'/alf/_ibl_fluo.times.npy')
            s1win = int(1/np.nanmedian(np.diff(fluo_times))) # How many frames to 1s (i.e. acquistion speed)
            DMS_autoc = autocorr(scipy.signal.medfilt(DMS, kernel_size=5))
            DLS_autoc = autocorr(scipy.signal.medfilt(DLS, kernel_size=5))
            NAcc_autoc = autocorr(scipy.signal.medfilt(NAcc, kernel_size=5))
            ses['DMS'] = DMS_autoc[:s1win]
            ses['DLS'] = DLS_autoc[:s1win]
            ses['NAcc'] = NAcc_autoc[:s1win]
            ses['DMS_signal_quality'] = (np.nanmax(DMS)/np.nanstd(DMS))>1
            ses['DLS_signal_quality'] = (np.nanmax(DLS)/np.nanstd(DLS))>1
            ses['NAcc_signal_quality'] = (np.nanmax(NAcc)/np.nanstd(NAcc))>1
            ses['auto_time'] = np.arange(0,1,(1/s1win))
            ses['DMS_signal_quality'] = (np.nanmax(DMS)/np.nanstd(DMS))>1
            ses['DLS_signal_quality'] = (np.nanmax(DLS)/np.nanstd(DLS))>1
            ses['NAcc_signal_quality'] = (np.nanmax(NAcc)/np.nanstd(NAcc))>1
            ses['ses'] = counter
            ses['mouse'] = path[-21:-15]
            data = pd.concat([data,ses])
            counter+=1
        except:
            print('Error in '+ str(path))    
    return data

def load_mouse_dataset_psth(MOUSE_FOLDER):
    mouse_folder = Path(MOUSE_FOLDER)
    data = pd.DataFrame()
    list_subfolders_with_paths = [f.path for f in os.scandir(mouse_folder) if f.is_dir()] 
    list_subfolders_with_paths.sort()
    counter=0
    for fld in list_subfolders_with_paths:
        try:
            ses = pd.DataFrame()
            path = fld+'/001'
            DMS = np.load(path+'/alf/_ibl_trials.DMS.npy')
            DLS = np.load(path+'/alf/_ibl_trials.DLS.npy')
            NAcc = np.load(path+'/alf/_ibl_trials.NAcc.npy')
            fluo_times = np.load(path+'/alf/_ibl_fluo.times.npy')
            s1win = int(1/np.nanmedian(np.diff(fluo_times))) # How many frames to 1s (i.e. acquistion speed)
            if s1win!=25: # Only analyzed data
                continue
            cue_times = np.load (path +'/alf/_ibl_trials.goCueTrigger_times.npy')
            feedback_times = np.load(path +'/alf/_ibl_trials.feedback_times.npy')[:-1]
            feedback = np.load(path +'/alf/_ibl_trials.feedbackType.npy')[:-1]
            fluo_in_trials_NAcc, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, NAcc, t_before_epoch = 0.1)
            fluo_in_trials_DLS, _ = divide_in_trials(fluo_times, cue_times, DLS, t_before_epoch = 0.1)
            fluo_in_trials_DMS, _ = divide_in_trials(fluo_times, cue_times, DMS, t_before_epoch = 0.1)
            rewarded_trials = np.where(feedback==1)[0]
            psth_NAcc,bins = fp_psth(fluo_time_in_trials, fluo_in_trials_NAcc, feedback_times, trial_list = rewarded_trials)
            psth_DLS,_ = fp_psth(fluo_time_in_trials, fluo_in_trials_DLS, feedback_times, trial_list = rewarded_trials)
            psth_DMS,_ = fp_psth(fluo_time_in_trials, fluo_in_trials_DMS, feedback_times, trial_list = rewarded_trials)
            psth_NAcc=np.nanmean(psth_NAcc, axis=0)
            psth_DLS=np.nanmean(psth_DLS, axis=0)
            psth_DMS=np.nanmean(psth_DMS, axis=0)
            ses['bins'] = bins
            ses['DMS'] = psth_DMS
            ses['DLS'] = psth_DLS
            ses['NAcc'] = psth_NAcc
            ses['DMS_signal_quality'] = (np.nanmax(DMS)/np.nanstd(DMS))>1
            ses['DLS_signal_quality'] = (np.nanmax(DLS)/np.nanstd(DLS))>1
            ses['NAcc_signal_quality'] = (np.nanmax(NAcc)/np.nanstd(NAcc))>1
            ses['ses'] = counter
            ses['mouse'] = path[-21:-15]
            data = pd.concat([data,ses])
            counter+=1
        except:
            print('Error in '+ str(path))    
    return data

#######################################################################################################################
################################################### Autocorrelogram analysis ##########################################
#######################################################################################################################

mice = ['fip_13','fip_14','fip_15','fip_16','fip_17','fip_20','fip_21']
mice = ['fip_17']
dataset = pd.DataFrame()
for mouse in mice:
    mouse_data = load_mouse_dataset(DATA_FOLDER+mouse)
    dataset = pd.concat([dataset, mouse_data])

dataset1=pd.DataFrame()
for mouse in dataset.mouse.unique():
    sec1 = dataset.loc[dataset['mouse']==mouse]
    for ses in sec1.ses.unique():
        sec2= sec1.loc[sec1['ses']==ses]
        samples = len(sec2['DMS'])
        if samples == 25:
                dataset1 = pd.concat([dataset1,sec2])

e = []
ses_summary=pd.DataFrame()
for mouse in dataset.mouse.unique():
    sec1 = dataset.loc[dataset['mouse']==mouse]
    for ses in sec1.ses.unique():
        sec2= sec1.loc[sec1['ses']==ses]
        try:
            idms = np.interp(np.arange(0,1,0.001),  sec2['auto_time'], sec2['DMS'])
            idls = np.interp(np.arange(0,1,0.001),  sec2['auto_time'], sec2['DLS'])
            inacc = np.interp(np.arange(0,1,0.001),  sec2['auto_time'], sec2['NAcc'])
            sec2['t_DMS'] = np.where(idms<=0.37)[0][0]/1000
            sec2['t_DLS'] = np.where(idls<=0.37)[0][0]/1000
            sec2['t_NAcc'] = np.where(inacc<=0.37)[0][0]/1000
            if  np.where(idms<=0.37)[0][0]/1000< np.where(idls<=0.37)[0][0]/1000:
                e.append(mouse)
        except:
            continue
        ses_summary = pd.concat([ses_summary,sec2])


dataset_curated  = dataset1.loc[(dataset1['DMS_signal_quality']==True)&(dataset1['DLS_signal_quality']==True)&(dataset1['NAcc_signal_quality']==True)]


fig, ax  = plt.subplots(2,4)
plt.sca(ax[0,0])
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_17'], x='auto_time', y='DMS', color='dodgerblue',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_17'], x='auto_time', y='DLS', color='orange',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_17'], x='auto_time', y='NAcc', color='k',ci=68)
plt.title('fip_17')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')
plt.sca(ax[0,1])
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_20'], x='auto_time', y='DMS', color='dodgerblue',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_20'], x='auto_time', y='DLS', color='orange',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_20'], x='auto_time', y='NAcc', color='k',ci=68)
plt.title('fip_20')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')
plt.sca(ax[0,2])
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_21'], x='auto_time', y='DMS', color='dodgerblue',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_21'], x='auto_time', y='DLS', color='orange',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_21'], x='auto_time', y='NAcc', color='k',ci=68)
plt.title('fip_21')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')
plt.sca(ax[1,0])
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_13'], x='auto_time', y='DMS', color='dodgerblue',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_13'], x='auto_time', y='DLS', color='orange',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_13'], x='auto_time', y='NAcc', color='k',ci=68)
plt.title('fip_13')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')
plt.sca(ax[1,1])
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_14'], x='auto_time', y='DMS', color='dodgerblue',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_14'], x='auto_time', y='DLS', color='orange',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_14'], x='auto_time', y='NAcc', color='k',ci=68)
plt.title('fip_14')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')
plt.sca(ax[1,2])
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_15'], x='auto_time', y='DMS', color='dodgerblue',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_15'], x='auto_time', y='DLS', color='orange',ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_15'], x='auto_time', y='NAcc', color='k',ci=68)
plt.title('fip_15')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')
plt.sca(ax[1,3])
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_16'], x='auto_time', y='DMS', color='dodgerblue', ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_16'], x='auto_time', y='DLS', color='orange', ci=68)
sns.lineplot(data = dataset_curated.loc[dataset_curated['mouse']=='fip_16'], x='auto_time', y='NAcc', color='k', ci=68)
plt.title('fip_16')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')


# Paired time constant
s1=ses_summary.melt(['mouse','ses'])
s1  = s1.loc[np.isin(s1.variable, ['t_DLS','t_DMS'])]
fig, ax  = plt.subplots(2,4, sharex =True, sharey=True)
plt.sca(ax[0,0])
sns.pointplot(data = s1.loc[s1['mouse']=='fip_17'], y='value', x='variable',ci=68, hue='ses')
plt.legend().remove()
plt.ylabel('Time constant (s)')
plt.title('fip_17')
plt.sca(ax[0,1])
sns.pointplot(data = s1.loc[s1['mouse']=='fip_20'], y='value', x='variable',ci=68, hue='ses')
plt.legend().remove()
plt.ylabel('Time constant (s)')
plt.title('fip_20')
plt.sca(ax[0,2])
sns.pointplot(data = s1.loc[s1['mouse']=='fip_21'], y='value', x='variable',ci=68, hue='ses')
plt.legend().remove()
plt.ylabel('Time constant (s)')
plt.title('fip_21')
plt.sca(ax[1,0])
sns.pointplot(data = s1.loc[s1['mouse']=='fip_13'], y='value', x='variable',ci=68, hue='ses')
plt.legend().remove()
plt.ylabel('Time constant (s)')
plt.title('fip_13')
plt.sca(ax[1,1])
sns.pointplot(data = s1.loc[s1['mouse']=='fip_14'], y='value', x='variable',ci=68, hue='ses')
plt.legend().remove()
plt.ylabel('Time constant (s)')
plt.title('fip_14')
plt.sca(ax[1,2])
sns.pointplot(data = s1.loc[s1['mouse']=='fip_15'], y='value', x='variable',ci=68, hue='ses')
plt.legend().remove()
plt.ylabel('Time constant (s)')
plt.title('fip_15')
plt.sca(ax[1,3])
sns.pointplot(data = s1.loc[s1['mouse']=='fip_16'], y='value', x='variable',ci=68, hue='ses')
plt.legend().remove()
plt.ylabel('Time constant (s)')
plt.title('fip_16')






#######################################################################################################################
################################################### Reward response ##########################################
#######################################################################################################################
mice = ['fip_13','fip_14','fip_15','fip_16','fip_17','fip_20','fip_21']
dataset_psth = pd.DataFrame()
for mouse in mice:
    mouse_data_psth = load_mouse_dataset_psth(DATA_FOLDER+mouse)
    dataset_psth = pd.concat([dataset_psth, mouse_data_psth])

dataset_curated  = dataset_psth.loc[(dataset_psth['DMS_signal_quality']==True)&(dataset_psth['DLS_signal_quality']==True)]

fig, ax  = plt.subplots(7,5, sharex=True, sharey=True)
for i, mouse in enumerate(mice):
    select1 =  dataset_curated.loc[dataset_curated['mouse']==mouse]
    select1['DMS']=stats.zscore(select1['DMS'],nan_policy='omit')
    select1['DLS']=stats.zscore(select1['DLS'],nan_policy='omit')
    for j in np.arange(5):
        plt.sca(ax[i,j])
        sns.lineplot(data = select1.loc[np.isin(select1['ses'], np.arange(j*5,j*5+5))], x='bins', y='DMS', color='dodgerblue', ci=68)
        sns.lineplot(data = select1.loc[np.isin(select1['ses'], np.arange(j*5,j*5+5))], x='bins', y='DLS', color='orange', ci=68)
        plt.title(mouse)
        plt.xlabel('Time (s)')
        plt.ylabel('z-score DF/F')
        legend_elements = [Patch(facecolor='dodgerblue', edgecolor='k',
                                label='DMS'),
                        Patch(facecolor='orange', edgecolor='k',
                                label='DLS')]
        ax[i,j].legend(handles=legend_elements, loc='upper right')
