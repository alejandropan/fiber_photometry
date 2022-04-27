import numpy as np 
import pandas as pd
from scipy import interpolate

def get_unixtime(dt64):
    return dt64.astype('datetime64[ns]').astype('int')

def extract_screen_time (ses):
    source= ses + '/raw_behavior_data/_iblrig_stimPositionScreen.raw.csv'
    data = pd.read_csv(source, names=["contrast", "position", "time"])
    data['unix'] = get_unixtime(data['time'].to_numpy())/1000000000
    data['time_diff'] = data['unix'].diff()
    data['reward']  = data['contrast'].diff()<0
    data['stim']  = data['contrast'].diff()>0
    data['contrast_of_feedback'] = data['contrast'][data.loc[data['reward']==True].index-1]
    data['contrast_of_feedback'] =  data['contrast_of_feedback'].shift(1)
    stim_from_bonsai = data.loc[~np.isnan(data['contrast_of_feedback']), 'contrast_of_feedback']
    stims = np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastRight.npy')) + np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastLeft.npy'))
    stim_times = np.load(ses+'/alf/_ibl_trials.stimOnTrigger_times.npy') 
    if len(stim_times) != len(data.loc[data['reward']==True]): #First contrast is not send for some reason
        assert np.sum(abs(stims[1:] - stim_from_bonsai)) == 0
        stim_times = stim_times[1:]
    data['bpod_time'] = np.nan
    data.loc[data['stim']==True, 'bpod_time'] = stim_times
    f = interpolate.interp1d(data.loc[(data['stim']==True), 'unix'], 
                            data.loc[(data['stim']==True), 'bpod_time'], 
                            fill_value='extrapolate')
    data.iloc[:,-1] = f(data['unix'])
    return data


if __name__ == "__main__":
	ses = sys.argv[1]
    data = extract_screen_time(ses)
    data.to_csv(ses+'/alf/_ibl_trials.screen_info.csv')