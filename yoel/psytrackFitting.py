import os
# run this as its where utils lives
work_dir = "/mnt/bucket/labs/witten/yoel/python/scripts"
os.chdir(work_dir)

import numpy as np
import re
import psytrack as psy
import utils

# first need to load a dataset
base_path = "/jukebox/witten/Alex/Session_slection_fp_pilot"
path_info = utils.get_paths(base_path, "fip_")

# loading one session for one mouse for testing
test_data = utils.load_data(path_info["fip_1"][-2])

# need to get the choices in the correct range for psytrack as it only accepts
# choices \in {0, 1} or {1, 2}, currently choices are \in {-1, 1}
test_data["psyChoice"] = (test_data["choice"] + 1) / 2
seed = 323612

# total number of trials, discarding last trial
N = test_data["choice"].shape[0] - 1

# neural data
nacc_core = utils.bleach_correct(test_data["fluoloc2"], avg_window=60)
fluo_in_trials, fluo_time_in_trials = utils.divide_in_trials(test_data["fluotimes"],
                                                       test_data["goCueTrigger_times"],
                                                       nacc_core,
                                                       t_before_epoch = 0.1)


weights = {"bias": 1, "lc": 1, "rc": 1}
K = np.sum([weights[i] for i in weights.keys()])
hyper = {"sigInit": 2**4., "sigma": [2**-4.]*K, "sigDay": None}
optList = ["sigma"]


input_data = {"y": np.floor(test_data["psyChoice"][:-1]), 
	     "inputs": 
	     {"lc": np.nan_to_num(test_data["contrastLeft"])[:-1][:, np.newaxis],
		 "rc": np.nan_to_num(test_data["contrastRight"])[:-1][:, np.newaxis]}}

hyp, evd, wMode, hess_info = psy.hyperOpt(input_data, hyper, weights, optList)

# test
