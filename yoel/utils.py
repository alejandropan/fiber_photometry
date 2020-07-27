import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def get_paths(base_path, search_str):
    unfiltered_paths_gen = os.walk(base_path, topdown=True)
    filtered_paths, fip_names = [], []
    for pset in unfiltered_paths_gen:
        for item in pset:
            if search_str and "2020-" and "alf" in item and isinstance(item, str):
                fip_names.append(item.split("/")[-4])
                filtered_paths.append(item)
    fip_names = np.unique(fip_names)
    fips = {f:[] for f in np.unique(fip_names)}
    for item in filtered_paths:
        fips[item.split("/")[-4]].append(item)
    # need to sort the paths by date
    for key, vals in fips.items():
        kidx = np.argsort([re.search(r'\d{4}-\d{2}-\d{2}', v).group() for v in vals])
        fips[key] = [fips[key][i] for i in kidx]
    return fips

def load_data(base_path):
    files = [os.path.join(base_path, x) for x in os.listdir(base_path)
             if ".D" not in x and ".npy" in x]
    out = {}
    for f in files:
        if "fluo" in f and "times" not in f:
            fname_part = f.split(".")[0].split("_")[-1]
            out["fluo"+fname_part] = np.load(f)
        elif "fluo.times" in f:
            fname_part = "times"
            out["fluo"+fname_part] = np.load(f)
        else:
            fname_part = f.split(".")[-2]
            out[fname_part] = np.load(f)
    return out

def avgAcc(data):
    uvals = np.unique(data)
    if -1 in uvals and 1 in uvals:
        datavals = data + 1
    else:
        msg = "feedback values not recognized"
        raise Exception(msg)
    max_val = datavals.max()
    return (datavals / datavals.max()).mean()


def avgRewardGivenPredictor(reward_vol, side_contrast):
    side_vals = np.unique(side_contrast)
    out = {u:0 for u in side_vals}
    # get things into zero and one range
    reward_oz = 1/reward_vol.max() * reward_vol
    # calculate average reward given value of side contrast
    for key in out.keys():
        idx = np.where(side_contrast == key)[0]
        out[key] = reward_oz[idx].mean()
    return out


def avgFluoGivenReward(reward_vol, fluo_data, r_val):
    # get indices of rewarded trials
    ridx = np.where(reward_vol == r_val)[0]
    avg_fluo_r = np.zeros(len(ridx))
    for a_idx, r_idx in enumerate(ridx):
        avg_fluo_r[a_idx] = fluo_data[r_idx].mean()
    return avg_fluo_r


def avgFluoBoth(reward_vol, fluo_data, r_val, u_val):
    N = len(reward_vol)
    # col 0 is reward, 1 is not reward
    avgs = np.empty((N, 2))
    avgs[:] = np.nan
    ridx = np.where(reward_vol == r_val)[0]
    uidx = np.setdiff1d(np.arange(len(reward_vol)), ridx)
    for r_idx in ridx:
        avgs[r_idx, 0] = fluo_data[r_idx].mean()
    for u_idx in uidx:
        avgs[u_idx, 1] = fluo_data[u_idx].mean()
    return avgs

def adjustDict(predictor_dict, prefix):
    return {prefix + "_" + str(key) : val for key, val in predictor_dict.items()}

def makeAvgDset(path_info, region_keys, time_key, trigger_cue_key, reward_key):
    pars = {key:{"covars":[], "dop":[]} for key in path_info.keys()}
    for key, val in path_info.items():
        for f_path in val:
            data = load_data(f_path)
            l_trials = np.nan_to_num(data["contrastLeft"])
            r_trials = np.nan_to_num(data["contrastRight"])
            signed_contrast = r_trials - l_trials
            df_list, dop_res = [], {}

            for r_key in region_keys:
                neu_core = bleach_correct(data[r_key], avg_window=60)
                fluo_in_trials, fluo_time_trials = divide_in_trials(data[time_key],
                                                                data[trigger_cue_key],
                                                                neu_core,
                                                                t_before_epoch = 0.1)
                # NEED TO FIX THIS BELOW TO NOT DEPEND ON -1 INDEX
                avgs = avgFluoBoth(data[reward_key][:-1], fluo_in_trials, 3, 0)
                dop_res[r_key] = np.nanmean(avgs, axis=0)
                # dont need this just use signed contrast
                #avg_covars_l = avgRewardGivenPredictor(data[reward_key][:-1], l_trials[:-1])
                #avg_covars_r = avgRewardGivenPredictor(data[reward_key][:-1], r_trials[:-1])
                avg_side_covars = avgRewardGivenPredictor(data[reward_key][:-1], signed_contrast[:-1])
                df_list.append(pd.DataFrame([adjustDict(avg_side_covars, r_key)]))

            df = pd.concat(df_list, axis=1).T
            pars[key]["covars"].append(df)
            pars[key]["dop"].append(dop_res)

    return pars

# functions below were written by Alex Pan

def bleach_correct(neural_data, avg_window = 60):
    '''
    Correct for bleaching of gcamp across the session. Calculates
    DF/F
    Parameters
    ----------
    nacc: alf file with fluorescence values for the area of interest
    avg_window: time for sliding window to calculate F value
    '''
    # First calculate sliding window
    F = np.convolve(neural_data, np.ones((avg_window,))/avg_window, mode='same')
    return neural_data / F


def divide_in_trials(fluo_times, cue_times, neural_data, t_before_epoch = 0.1):
    '''
    Makes list of list with fluorescnece and timestamps divided by trials
    Deletes last trial since it might be incomplte
    Parameters
    ----------
    
    Fluo_times: alf file with times for fluorescence frames
    cue_times:  alf file with times for cue (the start of the trial)
    nacc: alf file with fluorescence values for the area of interest
    '''
    #Delete before first trial and last trial

    fluo_times_s = np.delete(fluo_times,np.where(fluo_times < cue_times[0]))
    neural_data_s = np.delete(neural_data, np.where(fluo_times < cue_times[0]))
    fluo_times_f = np.delete(fluo_times_s, np.where(fluo_times_s > cue_times[-1]))
    neural_data_f = np.delete(neural_data_s, np.where(fluo_times_s > cue_times[-1]))
    neural_data_f = stats.zscore(neural_data_f)
    fluo_in_trials = []
    fluo_time_in_trials = []

    for i in range(len(cue_times)-1):
        # Get from 0.3 before
        fluo_in_trials.append(neural_data_f[(fluo_times_f
                                         >= cue_times[i] - t_before_epoch) &
                   (fluo_times_f <= cue_times[i+1])])
        fluo_time_in_trials.append(fluo_times_f[(fluo_times_f
                                         >= cue_times[i] - t_before_epoch) &
                   (fluo_times_f <= cue_times[i+1])])

    return fluo_in_trials, fluo_time_in_trials



