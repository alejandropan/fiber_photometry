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

# need to use the above to predict the average modulation of reward response on 
# rewarded trials (and maybe also unrewarded trials)

# this averages the fluoresence within rewarded trials and returns 
# an array of those averages
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

# from here down I'm running things / no functions should be defined
dp = "/jukebox/witten/Alex/Session_slection_fp_pilot/fip_1/2020-01-16/001/alf/"
base_path = "/jukebox/witten/Alex/Session_slection_fp_pilot"


path_info = get_paths(base_path, "fip_")
accs = {key:0 for key in path_info.keys()}

for fname, pathset in path_info.items():
    accs[fname] = np.zeros(len(pathset))
    for idx, path in enumerate(pathset):
        data = load_data(path)
        accs[fname][idx] = avgAcc(data["feedbackType"])

plt.plot(accs["FIP1"])
plt.plot(accs["FIP2"])
plt.legend(["fip1", "fip2"])
plt.xlabel("session #")
plt.ylabel("average accuracy")
plt.savefig("averageAcc2Mice.png", bbox_inches="tight", dpi=200)
plt.close()

data = load_data(dp)
print(data.keys())

# example: 
l_trials = np.nan_to_num(data["contrastLeft"])
r_trials = np.nan_to_num(data["contrastRight"])
signed_contrast = r_trials - l_trials
nacc_core = bleach_correct(data["fluoloc2"], avg_window=60)

fluo_in_trials, fluo_time_in_trials = divide_in_trials(data["fluotimes"],
                                                       data["goCueTrigger_times"],
                                                       nacc_core,
                                                       t_before_epoch = 0.1)

# test
t = avgFluoBoth(data["rewardVolume"][:-1], fluo_in_trials, 3, 0)
# average them, 0=rewardedAverage, 1=unrewardedAverage
avgs = np.nanmean(t, axis=0)

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

# testing the above
path_info = get_paths(base_path, "fip_")
region_keys = ["fluoloc1", "fluoloc2", "fluoloc3", "fluoloc4"]
time_key = "fluotimes"
trigger_cue_key = "goCueTrigger_times"
reward_key = "rewardVolume"

test = makeAvgDset(path_info, region_keys, time_key, trigger_cue_key, reward_key)
# get the final dataframe of covariates
covarPerSession = pd.concat(test["fip_1"]["covars"], axis=1)
covarPerSession.columns = ["sess{}".format(i+1) for i in range(len(covarPerSession.columns))]
covarPerSession = covarPerSession.T

