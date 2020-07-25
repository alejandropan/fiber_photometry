import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dp = "/jukebox/witten/Alex/Session_slection_fp_pilot/fip_1/2020-01-16/001/alf/"
base_path = "/jukebox/witten/Alex/Session_slection_fp_pilot"

def get_paths(base_path):
    unfiltered_paths_gen = os.walk(base_path, topdown=True)
    filtered_paths, fip_names = [], []
    for pset in unfiltered_paths_gen:
	for item in pset:
	    if "fip_" and "2020-" and "alf" in item and isinstance(item, str):
		fip_names.append(item.split("/")[-4])
		filtered_paths.append(item)
    fip_names = np.unique(fip_names)
    fips = {f:[] for f in np.unique(fip_names)}
    for item in filtered_paths:
	fips[item.split("/")[-4]].append(item)
    return filtered_paths, fips


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

data = load_data(dp)
print(data.keys())

# example: 
l_trials = np.nan_to_num(data["contrastLeft"])
r_trials = np.nan_to_num(data["contrastRight"])
signed_contrast = r_trials - l_trials
nacc = bleach_correct(data["fluoloc3"], avg_window=60)

fluo_in_trials, fluo_time_in_trials = divide_in_trials(data["fluotimes"], 
                                                       data["goCueTrigger_times"], 
						       nacc, 
						       t_before_epoch = 0.1)
def avgAcc(data):
    uvals = np.unique(data)
    if -1 in uvals and 1 in uvals:
	datavals = data + 1
    else:
	msg = "feedback values not recognized"
        raise Exception(msg)
    max_val = datavals.max()
    return (datavals / datavals.max()).mean()

# write a function to automatically read in the correct paths,
# this won't work when data set grows
# THIS IS FOR PLOTTING AVG ACC ACROSS SESSIONS
path_info = get_paths(base_path)
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


def avgRewardGivenPredictor(reward_vol, side_contrast):
    side_vals = np.unique(side_contrast)
    out = {u:0 for u in side_vals}
    # get things into zero and one range
    max_r = reward_vol.max()
    reward_oz = -1/max_r * (reward_vol - max_r)
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


