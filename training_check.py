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
MICE =['DMS_1',
'DMS_2',
'DMS_3',
'DMS_4',
'DMS_5',
'DMS_6',
'DMS_7',
'DMS_8']

CONTRA_STIM = {'DMS_1':'R',
'DMS_2':'R',
'DMS_3':'L',
'DMS_4':'L',
'DMS_5':'L',
'DMS_6':'R',
'DMS_7':'R',
'DMS_8':'L'}

GROUP = {'DMS_1':'YFP',
'DMS_2':'ChRmine',
'DMS_3':'ChRmine',
'DMS_4':'ChRmine',
'DMS_5':'YFP',
'DMS_6':'YFP',
'DMS_7':'ChRmine',
'DMS_8':'YFP'}

data=pd.DataFrame()
for mouse in MICE:
    mouse_df = mouse_data_loader(ROOT+mouse,  extract=False)
    mouse_df['mouse'] = mouse
    data = pd.concat([data, mouse_df])
data['feedbackType'] = 1*(data['feedbackType']>0)
data = data.reset_index()


data['group'] = data.mouse.map(GROUP)
data['contra_stim'] = data.mouse.map(CONTRA_STIM)



# Performance
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.lineplot(data=data.loc[data['group']=='ChRmine'],x='day',y='feedbackType',hue='mouse', errorbar=None, palette='Reds')
sns.lineplot(data=data.loc[data['group']=='YFP'],x='day',y='feedbackType',hue='mouse', errorbar=None, palette='Greys')
plt.xlabel('Training Day')
plt.ylabel('Fraction Correct')
plt.ylim(0,1)
sns.despine()
plt.sca(ax[1])
sns.lineplot(data=data.loc[data['group']=='ChRmine'],x='day',y='response_times',hue='mouse', errorbar=None, palette='Reds', estimator=np.median)
sns.lineplot(data=data.loc[data['group']=='YFP'],x='day',y='response_times',hue='mouse', errorbar=None, palette='Greys', estimator=np.median)
plt.xlabel('Training Day')
plt.ylim(0, 2)
plt.ylabel('Median decision time')
sns.despine()


# Performance averged
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
count = data.groupby(['day','mouse','group']).count()['feedbackType'].reset_index()
sns.lineplot(data=count.loc[count['group']=='ChRmine'],x='day',y='feedbackType',hue='mouse', errorbar=None, palette='Reds')
sns.lineplot(data=count.loc[count['group']=='YFP'],x='day',y='feedbackType',hue='mouse', errorbar=None, palette='Greys')
plt.xlabel('Training Day')
plt.ylabel('Fraction Correct')
sns.despine()
plt.sca(ax[1])
count = data.groupby(['day','mouse','group']).mean()['feedbackType'].reset_index()
sns.lineplot(data=count.loc[count['group']=='ChRmine'],x='day',y='feedbackType', errorbar='se', color='r', estimator=np.median)
sns.lineplot(data=count.loc[count['group']=='YFP'],x='day',y='feedbackType', errorbar='se', color='grey', estimator=np.median)
plt.xlabel('Training Day')
plt.ylabel('Median decision time')
sns.despine()

# Trial number
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
rt = data.groupby(['day','mouse','group']).count()['response_times'].reset_index()
sns.lineplot(data=data.loc[data['group']=='ChRmine'],x='day',y='feedbackType', errorbar='se', color='r')
sns.lineplot(data=data.loc[data['group']=='YFP'],x='day',y='feedbackType', errorbar='se', color='grey')
plt.xlabel('Training Day')
plt.ylabel('Fraction Correct')
plt.ylim(0,1)
sns.despine()
plt.sca(ax[1])
rt = data.groupby(['day','mouse','group']).count()['response_times'].reset_index()
sns.lineplot(data=rt.loc[rt['group']=='ChRmine'],x='day',y='response_times', errorbar='se', color='r')
sns.lineplot(data=rt.loc[rt['group']=='YFP'],x='day',y='response_times', errorbar='se', color='grey')
plt.xlabel('Training Day')
plt.ylabel('Trial number')
sns.despine()

# Overall bias
bias = data.groupby(['day','mouse','contra_stim', 'group']).mean()['choice'].reset_index()
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.pointplot(data=bias.loc[(bias['group']=='ChRmine')&(bias['contra_stim']=='L')],x='day',y='choice',hue='mouse', errorbar=None, palette='Reds')
sns.pointplot(data=bias.loc[(bias['group']=='YFP')&(bias['contra_stim']=='L')],x='day',y='choice',hue='mouse', errorbar=None, palette='Greys')
plt.sca(ax[1])
sns.pointplot(data=bias.loc[(bias['group']=='ChRmine')&(bias['contra_stim']=='R')],x='day',y='choice',hue='mouse', errorbar=None, palette='Reds')
sns.pointplot(data=bias.loc[(bias['group']=='YFP')&(bias['contra_stim']=='R')],x='day',y='choice',hue='mouse', errorbar=None, palette='Greys')
sns.despine()


# Contra choice depending on stimulus
data['contra_choice'] = 0
data.loc[(data['choice']==1)&(data['contra_stim']=='R'), 'contra_choice']=1
data.loc[(data['choice']==-1)&(data['contra_stim']=='L'), 'contra_choice']=1
data['contra_screen'] = 0
data.loc[(data['signed_contrast']>0)&(data['contra_stim']=='R'), 'contra_screen']=1
data.loc[(data['signed_contrast']<0)&(data['contra_stim']=='L'), 'contra_screen']=1
contra_pref = data.groupby(['day','mouse','contra_screen', 'group']).mean()['contra_choice'].reset_index()
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine') & (contra_pref['contra_screen']==1)],x='day',y='contra_choice', errorbar='se', color='r')
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['contra_screen']==1)],x='day',y='contra_choice', errorbar='se', color='grey')
plt.xlabel('training day')
plt.ylabel('Fraction of contra choice')
plt.title('Contra stimulus')
plt.sca(ax[1])
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine') & (contra_pref['contra_screen']==0)],x='day',y='contra_choice', errorbar='se', color='r')
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['contra_screen']==0)],x='day',y='contra_choice', errorbar='se', color='grey')
plt.xlabel('training day')
plt.ylabel('Fraction of contra choice')
plt.title('Ipsi stimulus')
sns.despine()


fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.pointplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine') & (contra_pref['contra_screen']==1)],hue='mouse',x='day',y='contra_choice', errorbar='se', palette='Reds')
sns.pointplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['contra_screen']==1)],hue='mouse',x='day',y='contra_choice', errorbar='se', palette='Greys')
plt.xlabel('training day')
plt.ylabel('Fraction of contra choice')
plt.title('Contra stimulus')
plt.sca(ax[1])
sns.pointplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine') & (contra_pref['contra_screen']==0)],hue='mouse',x='day',y='contra_choice', errorbar='se', palette='Reds')
sns.pointplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['contra_screen']==0)],hue='mouse',x='day',y='contra_choice', errorbar='se', palette='Greys')
plt.xlabel('training day')
plt.ylabel('Fraction of contra choice')
plt.title('Ipsi stimulus')
sns.despine()



# contra choices after contra
data['after_contra_stim'] = data['contra_screen'].shift(1)
contra_pref = data.groupby(['day','mouse','after_contra_stim', 'group']).mean()['contra_choice'].reset_index()
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine') & (contra_pref['after_contra_stim']==1)],x='day',y='contra_choice', errorbar='se', color='r')
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['after_contra_stim']==1)],x='day',y='contra_choice', errorbar='se', color='grey')
plt.xlabel('training day')
plt.ylabel('Fraction of contra choice')
plt.title('After contra stimulus')
plt.sca(ax[1])
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine') & (contra_pref['after_contra_stim']==0)],x='day',y='contra_choice', errorbar='se', color='r')
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['after_contra_stim']==0)],x='day',y='contra_choice', errorbar='se', color='grey')
plt.xlabel('training day')
plt.ylabel('Fraction of contra choice')
plt.title('After ipsi stimulus')
sns.despine()


# RTs 
col_ipsi = ["#94D2BD", "#0A9396","#005F73", "#001219"]
col_contra = ["#F48C06", "#DC2F02", "#9D0208", "#370617"]

data['abs_contrast'] = abs(data['signed_contrast'])


contra_pref = data.groupby(['day','mouse','contra_screen', 'group']).median()['response_times'].reset_index()
fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine')& (contra_pref['contra_screen']==1)],x='day',y='response_times', errorbar='se', color='r', estimator=np.median)
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['contra_screen']==1)],x='day',y='response_times', errorbar='se', color='grey',estimator=np.median)
plt.xlabel('Training Day')
plt.ylabel('Median decision time')
plt.title('Contra trials')
plt.ylim(0,2)
sns.despine()
plt.sca(ax[1])
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='ChRmine')& (contra_pref['contra_screen']==0)],x='day', y='response_times', errorbar='se', color='r', estimator=np.median)
sns.lineplot(data=contra_pref.loc[(contra_pref['group']=='YFP') & (contra_pref['contra_screen']==0)],x='day', y='response_times', errorbar='se', color='grey',estimator=np.median)
plt.xlabel('Training Day')
plt.ylim(0, 2)
plt.ylabel('Median decision time')
plt.title('Ipsi trials')
sns.despine()