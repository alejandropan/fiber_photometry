from logging import PlaceHolder
from tkinter.ttk import Style
import pandas as pd
import numpy as np
from pathlib import Path
import os
from ibllib.io.extractors.biased_trials import extract_all
import one.alf as alf
from ibllib.io.raw_data_loaders import load_settings
import zipfile
from ibllib.io.extractors.training_trials import (
    Choice, FeedbackTimes, FeedbackType, GoCueTimes, Intervals, ItiDuration, ProbabilityLeft, ResponseTimes, RewardVolume, StimOnTimes_deprecated)
from ibllib.io.extractors.biased_trials import ContrastLR

def alf_loader(alfpath):
    data=pd.DataFrame()
    data['choice'] = np.load(alfpath+'/_ibl_trials.choice.npy')*-1
    data['feedbackType'] = np.load(alfpath+ '/_ibl_trials.feedbackType.npy')
    data['response_times'] = np.load(alfpath+ '/_ibl_trials.response_times.npy') - np.load(alfpath+ '/_ibl_trials.goCueTrigger_times.npy')
    data['contrastRight'] = np.load(alfpath+ '/_ibl_trials.contrastRight.npy')
    data['contrastLeft'] = np.load(alfpath+ '/_ibl_trials.contrastLeft.npy')
    data.loc[np.isnan(data['contrastRight']), 'contrastRight'] = 0
    data.loc[np.isnan(data['contrastLeft']), 'contrastLeft'] = 0
    data['signed_contrast'] = data['contrastRight'] - data['contrastLeft']
    return data

def mouse_data_loader(rootdir, extract=True):
    '''
    rootdir (str): mouse directory
    variables (list): list containing the keys of the variables of interest
    Will extract and load data from the whole life of animal
    '''
    mouse_df = pd.DataFrame()
    counter = 0
    for file in sorted(os.listdir(rootdir)):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            print(d)
            day_df = pd.DataFrame()
            counter += 1
            for ses in sorted(os.listdir(d)):
                s = os.path.join(d, ses)
                if os.path.isdir(s):
                    try:
                        if extract==True:
                            extract_all(
                                session_path=s, save=True, extra_classes=[Intervals, FeedbackType, ProbabilityLeft, Choice, ItiDuration,
                                StimOnTimes_deprecated, RewardVolume, FeedbackTimes, ResponseTimes, GoCueTimes,ContrastLR])
                        elif Path(s+'/alf').is_dir()==False:
                            extract_all(
                                session_path=s, save=True, extra_classes=[Intervals, FeedbackType, ProbabilityLeft, Choice, ItiDuration,
                                StimOnTimes_deprecated, RewardVolume, FeedbackTimes, ResponseTimes, GoCueTimes,ContrastLR])
                        ses_df= alf_loader(s+'/alf')
                        day_df = pd.concat([day_df,ses_df])
                    except:
                        continue
            day_df['day'] = counter
            mouse_df = pd.concat([mouse_df,day_df])
    return mouse_df


ROOT = '/Volumes/witten/Alex/Data/Subjects/'
MICE = ['fip_34','fip_35','fip_36','fip_37','fip_38']
data=pd.DataFrame()
for mouse in MICE:
    mouse_df = mouse_data_loader(ROOT+mouse,  extract=False)
    mouse_df['mouse'] = mouse
    data = pd.concat([data, mouse_df])
data['feedbackType'] = 1*(data['feedbackType']>0)
data = data.reset_index()

fig,ax = plt.subplots(1,2)
# Analysis 100_0 step
plt.sca(ax[0])
sns.lineplot(data=data,x='day',y='feedbackType',hue='mouse', errorbar=None)
plt.xlabel('Training Day')
plt.ylabel('Fraction Correct')
plt.ylim(0,1)
sns.despine()
plt.sca(ax[1])
sns.lineplot(data=data,x='day',y='response_times',hue='mouse', errorbar='se')
plt.xlabel('Training Day')
plt.ylabel('Response Time')
sns.despine()
