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
import seaborn as sns
import matplotlib.pyplot as plt

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

def mouse_data_loader(rootdir, extract=False):
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
MICE =['fip_13',
'fip_14',
'fip_15',
'fip_16',
'fip_26',
'fip_27',
'fip_28',
'fip_29',
'fip_30',
'fip_31',
'fip_32',
'fip_33',
'fip_34',
'fip_35',
'fip_36',
'fip_37',
'fip_38',
'fip_39',
'fip_40',
'fip_41',
'fip_42',
'fip_43'
]


GROUP ={'fip_13':'M',
'fip_14':'M',
'fip_15':'M',
'fip_16':'M',
'fip_26':'M',
'fip_27':'M',
'fip_28':'M',
'fip_29':'M',
'fip_30':'M',
'fip_31':'M',
'fip_32':'M',
'fip_33':'M',
'fip_34':'F',
'fip_35':'F',
'fip_36':'F',
'fip_37':'F',
'fip_38':'F',
'fip_39':'M',
'fip_40':'M',
'fip_41':'F',
'fip_42':'F',
'fip_43':'F',
}

INITIAL_WEIGHTS ={'fip_13':21.3,
'fip_14':21,
'fip_15':21,
'fip_16':21,
'fip_26':22,
'fip_27':22,
'fip_28':24,
'fip_29':22,
'fip_30':23,
'fip_31':23,
'fip_32':23,
'fip_33':26,
'fip_34':20.5,
'fip_35':21.6,
'fip_36':21.3,
'fip_37':20,
'fip_38':19,
'fip_39':21.4,
'fip_40':19.5,
'fip_41':16.6,
'fip_42':16.1,
'fip_43':15.3,
}

data=pd.DataFrame()
for mouse in MICE:
    mouse_df = mouse_data_loader(ROOT+mouse,  extract=False)
    mouse_df['mouse'] = mouse
    data = pd.concat([data, mouse_df])
data['feedbackType'] = 1*(data['feedbackType']>0)
data = data.reset_index()

data['group'] = data.mouse.map(GROUP)


# Add trials to date
data['trials_to_date'] = 0
for mouse in MICE:
    data.loc[data['mouse']==mouse, 'trials_to_date'] = \
        np.arange(len(data.loc[data['mouse']==mouse, 'trials_to_date']))

data_easy = data.loc[abs(data['signed_contrast'])==1]
summary_learning = data_easy.groupby(['mouse','day']).mean()['feedbackType'].reset_index()
summary_learning['trials_to_date'] = data_easy.groupby(['mouse','day']).max().reset_index()['trials_to_date'].to_numpy()

good_days = summary_learning.loc[summary_learning['feedbackType']>=0.8]
learning_days = good_days.groupby(['mouse']).min()[['day','trials_to_date']].reset_index()

learning_days=learning_days.reset_index()
learning_days['group']=learning_days.mouse.map(GROUP)

mouse_that_didnt_learn = np.setxor1d(learning_days['mouse'].unique(),MICE)

didntlearn= pd.DataFrame()
didntlearn['mouse'] = mouse_that_didnt_learn
didntlearn['group'] = didntlearn.mouse.map(GROUP)
didntlearn['day'] = np.nan
didntlearn['trials_to_date'] = np.nan

# Summary plots
fig,ax = plt.subplots(1,3)
plt.sca(ax[0])
sns.swarmplot(data = learning_days, x = 'group', y='day', color='k')
sns.barplot(data = learning_days, x = 'group', y='day')
plt.xlabel('Sex')
plt.ylabel('Learning Day')
plt.sca(ax[1])
sns.swarmplot(data = learning_days, x = 'group', y='trials_to_date',  color='k')
sns.barplot(data = learning_days, x = 'group', y='trials_to_date')
plt.xlabel('Sex')
plt.ylabel('Learning Trial')
plt.sca(ax[2])
sns.barplot(data = didntlearn.groupby(['group']).count().reset_index(), x = 'group', y='mouse')
plt.ylabel('Number of mice')
plt.xlabel('Sex')
sns.despine()

learning_days['learned'] = 'Learned'
didntlearn['learned'] = 'did not learn'
learning =pd.concat([learning_days, didntlearn])
learning['iw'] = learning.mouse.map(INITIAL_WEIGHTS)
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.swarmplot(data = learning, x = 'group', y='iw', color='k')
sns.barplot(data = learning, x = 'group', y='iw', color='gray')
plt.xlabel('Sex')
plt.ylabel('Initial Weight')
plt.sca(ax[1])
sns.barplot(data = learning, x = 'group', y='iw', hue='learned')
plt.xlabel('Sex')
plt.ylabel('Initial Weight')
sns.despine()


