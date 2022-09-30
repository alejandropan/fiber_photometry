import numpy as np 
import pandas as pd
from scipy import interpolate
import sys
from matplotlib import pyplot as plt

def get_unixtime(dt64):
    return dt64.astype('datetime64[ns]').astype('int')

def extract_screen_time(ses, save=True):
    source= ses + '/raw_behavior_data/_iblrig_stimPositionScreen.raw.csv'
    data = pd.read_csv(source, names=["contrast", "position", "time"])
    data['unix'] = get_unixtime(data['time'].to_numpy())/1000000000
    data['time_diff'] = data['unix'].diff()
    data['reward']  = data['contrast'].diff()<0
    data['stim']  = data['contrast'].diff()>0
    data['contrast_of_feedback'] = data['contrast'][data.loc[data['reward']==True].index-1]
    data['contrast_of_feedback'] =  data['contrast_of_feedback'].shift(1)
    stim_from_bonsai = data.loc[~np.isnan(data['contrast_of_feedback']), 'contrast_of_feedback'].to_numpy()
    stims = np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastRight.npy')) + np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastLeft.npy'))
    stim_times = np.load(ses+'/alf/_ibl_trials.stimOnTrigger_times.npy') 
    if len(stims[1:]) != len(data.loc[data['stim']==True]): #First contrast is not send for some reason
        assert np.array_equal(stims[1:], data.loc[data['stim']==True, 'contrast'].to_numpy()[:-1])
        data['bpod_time'] = np.nan
        data.loc[data['stim']==True, 'bpod_time'] = np.concatenate([stim_times[1:],np.array([np.nan])])
    else:
        assert np.array_equal(stims[1:], data.loc[data['stim']==True, 'contrast'].to_numpy())
        data['bpod_time'] = np.nan
        data.loc[data['stim']==True, 'bpod_time'] = stim_times[1:]
    #data.loc[data['stim']==True, 'bpod_time'] = stim_times #This doesnt work?
    f = interpolate.interp1d(data.loc[(data['stim']==True), 'unix'], 
                            data.loc[(data['stim']==True), 'bpod_time'], 
                            fill_value='extrapolate')
    data.iloc[:,-1] = f(data['unix'])
    if len(stims[1:]) != len(data.loc[data['stim']==True]): #First contrast is not send for some reason
        error = np.median(abs(np.diff(data.loc[data['stim']==True, 'unix'].to_numpy()[:-1]) - np.diff(stim_times[1:])))
    else:
        error = np.median(abs(np.diff(data.loc[data['stim']==True, 'unix'].to_numpy()) - np.diff(stim_times[1:])))
    assert error < 0.010
    if save==True:
        data.to_csv(ses+'/alf/_ibl_trials.screen_info.csv')
    return data

def plot_screen_position_sync(ses):
    stim_times = np.load(ses+'/alf/_ibl_trials.stimOnTrigger_times.npy')
    stims = 35*(np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastRight.npy')) - np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastLeft.npy'))>0)
    stims[np.where(stims==0)] = -35
    fed_type = np.load(ses+'/alf/_ibl_trials.feedbackType.npy')
    fed = np.load(ses+'/alf/_ibl_trials.feedback_times.npy')
    data = pd.read_csv(ses+'/alf/_ibl_trials.screen_info.csv')
    wheel_pos = np.load(ses+'/alf/_ibl_wheel.position.npy') * 124.017 * -1 # Convert to degrees and invert (wheel and patch move on opposite direction)
    wheel_times = np.load(ses+'/alf/_ibl_wheel.timestamps.npy')
    for i in np.arange(len(stim_times)):
        wheel_pos[np.where((wheel_times>=stim_times[i]))] = \
                wheel_pos[np.where((wheel_times>=stim_times[i]))] - \
                wheel_pos[np.where((wheel_times>=stim_times[i]))][0] + \
                stims[i]
    screen_time = np.arange(data['bpod_time'].min(), 
                        data['bpod_time'].max(),0.001)
    screen_pos = np.interp(screen_time,data['bpod_time'],data['position'])
    screen_contrast = np.interp(screen_time, data['bpod_time'], data['contrast'])
    screen_pos = screen_pos[np.where(screen_contrast>0)]
    screen_time = screen_time[np.where(screen_contrast>0)]
    plt.scatter(screen_time, screen_pos, s=1)
    plt.plot(wheel_times, wheel_pos, linestyle='dashed', color='orange')
    plt.vlines(fed[np.where(fed_type==1)], -70,70, color='g')
    plt.vlines(fed[np.where(fed_type==-1)], -70,70, color='r')
    plt.vlines(stim_times, -70,70, color='k')
    plt.hlines(0,0,stim_times[-1],linestyles='dashed', color='gray')
    plt.hlines(35,0,stim_times[-1],linestyles='dashed', color='gray')
    plt.hlines(-35,0,stim_times[-1],linestyles='dashed', color='gray')
    plt.hlines(70,0,stim_times[-1],linestyles='dashed', color='gray')
    plt.hlines(-70,0,stim_times[-1],linestyles='dashed', color='gray')
    plt.xlabel('Time (s)')
    plt.ylabel('Patch position (deg)')
    plt.show()


if __name__ == "__main__":
    ses = sys.argv[1]
    data = extract_screen_time(ses)
    #plot_screen_position_sync(ses)