# Check delays between response and feedback for bad trials
import os
from pathlib import Path
import numpy as np
import pandas as pd
from psths import divide_in_trials
import seaborn as sns

MOUSE_FOLDER = '/Volumes/witten/Alex/Data/Subjects/fip_20'
def load_mouse_dataset(MOUSE_FOLDER):
    mouse_folder = Path(MOUSE_FOLDER)
    data = pd.DataFrame()
    list_subfolders_with_paths = [f.path for f in os.scandir(mouse_folder) if f.is_dir()] 
    for fld in list_subfolders_with_paths:
        try:
            ses = pd.DataFrame()
            path = fld+'/001'
            ses['choice']  = -1*(np.load(path+'/alf/_ibl_trials.choice.npy'))
            ses['outcome']  = (np.load(path+'/alf/_ibl_trials.feedbackType.npy')>0)*1
            ses['probabilityLeft']  = np.load(path+'/alf/_ibl_trials.probabilityLeft.npy')
            ses['goCue_trigger_times'] = np.load(path+'/alf/_ibl_trials.goCue_times.npy')
            ses['stimOn_times'] = np.load(path+'/alf/_ibl_trials.stimOn_times.npy')
            ses['response_times'] = np.load(path+'/alf/_ibl_trials.response_times.npy')
            ses['feedback_times'] = np.load(path+'/alf/_ibl_trials.feedback_times.npy')
            fluo,_ = divide_in_trials(np.load(path+'/alf/_ibl_fluo.times.npy'),
                                          np.load(path+'/alf/_ibl_trials.goCue_times.npy'),
                                          np.load(path+'/alf/_ibl_trials.DMS.npy'))
            DMS=[]
            for i in fluo:
                DMS.append(np.mean(i))
            DMS.append(np.nan) # Toaccount for divide in trial deleting the last trial
            ses['DMS'] = DMS
            wheel = np.load(path+'/alf/_ibl_wheel.position.npy')
            wheel_times = np.load(path+'/alf/_ibl_wheel.timestamps.npy')
            t_start = []
            for i in ses['goCue_trigger_times']:
                diff= np.absolute(wheel_times-i)
                t_start.append(diff.argmin())
            ses['wheel_position'] = wheel[t_start]
            data = pd.concat([data,ses])
        except:
            print('Error in '+ str(path))    
    return data


fig, ax = plt.subplots(2)
plt.sca(ax[0])
data = load_mouse_dataset(MOUSE_FOLDER)
sns.scatterplot(data=data.loc[~np.isnan(data['DMS'])], y='DMS', x='wheel_position')
plt.xlabel('Wheel position')
plt.ylabel('DF/F')
plt.sca(ax[1])
sns.scatterplot(data=data.loc[~np.isnan(data['DMS'])], y='DMS', x='goCue_trigger_times')
plt.xlabel('Trial start time')
plt.ylabel('DF/F')

data['delta'] = data['feedback_times']-data['response_times']
sns.histplot(data=data.loc[data['outcome']==0], x='delta', stat='probability')
plt.xlabel('Choice to noise s')
plt.ylabel('Fraction')
plt.title('Fip 17')