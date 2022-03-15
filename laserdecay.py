from psths import *
import pandas as pd
import seaborn as sns
from scipy.stats import zscore

ses = '/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-09/001'
def analysis_decay_non_opto(ses, subtract_baseline = False):
    fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
    NAcc = np.load(ses +'/alf/_ibl_trials.NAcc.npy')
    DLS = np.load(ses +'/alf/_ibl_trials.DLS.npy')
    DMS = np.load(ses +'/alf/_ibl_trials.DMS.npy')
    cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
    response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
    feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
    left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
    right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
    l_trials = np.nan_to_num(left_trials)
    r_trials = np.nan_to_num(right_trials)
    signed_contrast = r_trials - l_trials
    feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
    choice = np.load(ses + '/alf/_ibl_trials.choice.npy')[:-1]
    response_times = response_times[:-1]
    feedback_times = feedback_times[:-1]
    l_trials = l_trials[:-1]
    r_trials = r_trials[:-1]
    signed_contrast = signed_contrast[:-1]
    feedback = feedback[:-1]
    stimOn_times  = np.load(ses + '/alf/_ibl_trials.stimOn_times.npy')

    fig, ax = plt.subplots(1,3, sharey=True)
    plt.sca(ax[0])
    fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, DLS, t_before_epoch = 0.2)
    for i in np.arange(3):
        if i==0:
            correct = np.where(feedback[:(i+1)*200] == 1)[0]
        else:
            correct = np.where(feedback == 1)[0]       
            correct = correct[(correct>=i*200) & (correct<(i+1)*200)]

        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = correct, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
        plot_psth(condition1, color='dodgerblue', alpha=1- 1/3*i, plot_error=False)
    plt.title('DLS')
    plt.sca(ax[1])
    fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, DMS, t_before_epoch = 0.2)
    for i in np.arange(3):
        if i==0:
            correct = np.where(feedback[:(i+1)*200] == 1)[0]
        else:
            correct = np.where(feedback == 1)[0]       
            correct = correct[(correct>=i*200) & (correct<(i+1)*200)]

        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = correct, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
        plot_psth(condition1, color='dodgerblue', alpha=1- 1/3*i, plot_error=False)
    plt.title('DMS')
    plt.sca(ax[2])
    fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, NAcc, t_before_epoch = 0.2)
    for i in np.arange(3):
        if i==0:
            correct = np.where(feedback[:(i+1)*200] == 1)[0]
        else:
            correct = np.where(feedback == 1)[0]       
            correct = correct[(correct>=i*200) & (correct<(i+1)*200)]

        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = correct, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
        plot_psth(condition1, color='dodgerblue', alpha=1- 1/3*i, plot_error=False)
    plt.title('NAcc')

    data = pd.DataFrame()
    data['choices'] = choice
    data['feedback'] = feedback
    data['repeat'] = 1*(data['choices'] == data['choices'].shift(1))
    data['feedback'] =  data['feedback'].shift(1)

    data =  data.reset_index()
    fig, ax = plt.subplots(1,3, sharey=True)
    plt.sca(ax[0])
    sns.barplot(data=data.loc[data['index']<200], y='repeat', x='feedback', palette='viridis', ci=68)
    plt.xlabel('Trial Outcome')
    plt.ylabel('Fraction of repeated choices')
    plt.title('Trial 1-199')
    plt.sca(ax[1])
    sns.barplot(data=data.loc[(data['index']>=200) & (data['index']<399)], y='repeat', x='feedback', palette='viridis', ci=68)
    plt.xlabel('Trial Outcome')
    plt.title('Trial 200-399')
    plt.sca(ax[2])
    sns.barplot(data=data.loc[(data['index']>=400) & (data['index']<600)], y='repeat', x='feedback', palette='viridis', ci=68)
    plt.xlabel('Opto Block')
    plt.title('Trial 400-599')
    plt.show()
def analysis_decay_opto(ses, subtract_baseline = False):
    fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
    NAcc = np.load(ses +'/alf/_ibl_trials.NAcc.npy')
    DLS = np.load(ses +'/alf/_ibl_trials.DLS.npy')
    DMS = np.load(ses +'/alf/_ibl_trials.DMS.npy')
    cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
    opto_block = np.load (ses +'/alf/_ibl_trials.opto_block.npy')
    response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
    feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
    left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
    right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
    l_trials = np.nan_to_num(left_trials)
    r_trials = np.nan_to_num(right_trials)
    signed_contrast = r_trials - l_trials
    feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
    choice = np.load(ses + '/alf/_ibl_trials.choice.npy')[:-1]
    opto_block = opto_block[:-1]
    response_times = response_times[:-1]
    feedback_times = feedback_times[:-1]
    l_trials = l_trials[:-1]
    r_trials = r_trials[:-1]
    signed_contrast = signed_contrast[:-1]
    feedback = feedback[:-1]
    stimOn_times  = np.load(ses + '/alf/_ibl_trials.stimOn_times.npy')

    fig, ax = plt.subplots(1,3, sharey=True)
    plt.sca(ax[0])
    fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, DLS, t_before_epoch = 0.2)
    for i in np.arange(3):
        if i==0:
            opto = np.where(opto_block[:(i+1)*200] == 1)[0]
            reward = np.where(opto_block[:(i+1)*200] == 0)[0]
            correct = np.where(feedback[:(i+1)*200] == 1)[0]
        else:
            opto = np.where(opto_block== 1)[0]
            reward = np.where(opto_block == 0)[0]
            correct = np.where(feedback == 1)[0]       
            opto = opto[(opto>=i*200) & (opto<(i+1)*200)]
            reward = reward[(reward>=i*200) & (reward<(i+1)*200)]
            correct = correct[(correct>=i*200) & (correct<(i+1)*200)]

        opto = np.intersect1d(opto, correct)
        reward = np.intersect1d(reward, correct)
        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = opto, t_range = [-0.2, 3.0])
        condition2 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = reward, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
            condition2 = (condition2[0]-F2, condition2[1])
        plot_psth(condition1, color='orange', alpha=1- 1/3*i, plot_error=False)
        plot_psth(condition2, color='dodgerblue', alpha=1- 1/3*i, plot_error=False)
    plt.title('DLS')
    plt.sca(ax[1])
    fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, DMS, t_before_epoch = 0.2)
    for i in np.arange(3):
        if i==0:
            opto = np.where(opto_block[:(i+1)*200] == 1)[0]
            reward = np.where(opto_block[:(i+1)*200] == 0)[0]
            correct = np.where(feedback[:(i+1)*200] == 1)[0]
        else:
            opto = np.where(opto_block== 1)[0]
            reward = np.where(opto_block == 0)[0]
            correct = np.where(feedback == 1)[0]       
            opto = opto[(opto>=i*200) & (opto<(i+1)*200)]
            reward = reward[(reward>=i*200) & (reward<(i+1)*200)]
            correct = correct[(correct>=i*200) & (correct<(i+1)*200)]

        opto = np.intersect1d(opto, correct)
        reward = np.intersect1d(reward, correct)
        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = opto, t_range = [-0.2, 3.0])
        condition2 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = reward, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
            condition2 = (condition2[0]-F2, condition2[1])
        plot_psth(condition1, color='orange', alpha=1- 1/3*i, plot_error=False)
        plot_psth(condition2, color='dodgerblue', alpha=1- 1/3*i, plot_error=False)
    plt.title('DMS')
    plt.sca(ax[2])
    fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, NAcc, t_before_epoch = 0.2)
    for i in np.arange(3):
        if i==0:
            opto = np.where(opto_block[:(i+1)*200] == 1)[0]
            reward = np.where(opto_block[:(i+1)*200] == 0)[0]
            correct = np.where(feedback[:(i+1)*200] == 1)[0]
        else:
            opto = np.where(opto_block== 1)[0]
            reward = np.where(opto_block == 0)[0]
            correct = np.where(feedback == 1)[0]       
            opto = opto[(opto>=i*200) & (opto<(i+1)*200)]
            reward = reward[(reward>=i*200) & (reward<(i+1)*200)]
            correct = correct[(correct>=i*200) & (correct<(i+1)*200)]

        opto = np.intersect1d(opto, correct)
        reward = np.intersect1d(reward, correct)

        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = opto, t_range = [-0.2, 3.0])
        condition2 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            feedback_times, trial_list = reward, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
            condition2 = (condition2[0]-F2, condition2[1])
        plot_psth(condition1, color='orange', alpha=1- 1/3*i, plot_error=False)
        plot_psth(condition2, color='dodgerblue', alpha=1- 1/3*i, plot_error=False)
    plt.title('NAcc')

    data = pd.DataFrame()
    data['choices'] = choice
    data['opto_block'] = opto_block
    data['feedback'] = feedback
    data['repeat'] = 1*(data['choices'] == data['choices'].shift(1))
    data['feedback'] =  data['feedback'].shift(1)

    data =  data.reset_index()
    fig, ax = plt.subplots(1,3, sharey=True)
    plt.sca(ax[0])
    sns.barplot(data=data.loc[data['index']<200], x = 'opto_block', y='repeat', hue='feedback', palette='viridis', ci=68)
    plt.xlabel('Trial Outcome')
    plt.ylabel('Fraction of repeated choices')
    plt.title('Trial 1-199')
    plt.sca(ax[1])
    sns.barplot(data=data.loc[(data['index']>=200) & (data['index']<399)], x = 'opto_block', y='repeat', hue='feedback', palette='viridis', ci=68)
    plt.xlabel('Trial Outcome')
    plt.title('Trial 200-399')
    plt.sca(ax[2])
    sns.barplot(data=data.loc[(data['index']>=400) & (data['index']<600)], x = 'opto_block', y='repeat', hue='feedback', palette='viridis', ci=68)
    plt.xlabel('Opto Block')
    plt.title('Trial 400-599')
    plt.show()
def analysis_decay_opto_summary(ses, subtract_baseline = False):
    fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
    NAcc = np.load(ses +'/alf/_ibl_trials.NAcc.npy')
    DLS = np.load(ses +'/alf/_ibl_trials.DLS.npy')
    DMS = np.load(ses +'/alf/_ibl_trials.DMS.npy')
    cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
    opto_block = np.load (ses +'/alf/_ibl_trials.opto_block.npy')
    response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
    feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
    left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
    right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
    l_trials = np.nan_to_num(left_trials)
    r_trials = np.nan_to_num(right_trials)
    signed_contrast = r_trials - l_trials
    feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
    choice = np.load(ses + '/alf/_ibl_trials.choice.npy')[:-1]
    opto_block = opto_block[:-1]
    response_times = response_times[:-1]
    feedback_times = feedback_times[:-1]
    l_trials = l_trials[:-1]
    r_trials = r_trials[:-1]
    signed_contrast = signed_contrast[:-1]
    feedback = feedback[:-1]
    stimOn_times  = np.load(ses + '/alf/_ibl_trials.stimOn_times.npy')
    reg_names = ['DLS','DMS','NAcc']

    _, ax = plt.subplots(2,3, sharey=True)
    for i, reg in enumerate([DLS,DMS, NAcc]):
        plt.sca(ax[0,i])
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, zscore(reg,nan_policy='omit'), t_before_epoch = 0.2)
        opto = np.where(opto_block == 1)[0]
        reward = np.where(opto_block == 0)[0]
        correct = np.where(feedback == 1)[0]
        opto = np.intersect1d(opto, correct)
        reward = np.intersect1d(reward, correct)
        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = opto, t_range = [-0.2, 3.0])
        condition2 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = reward, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
            condition2 = (condition2[0]-F2, condition2[1])
        plot_psth(condition1, color='orange', plot_error=True)
        plot_psth(condition2, color='dodgerblue', plot_error=True)
        plt.title(reg_names[i])

    for i, reg in enumerate([DLS,DMS, NAcc]):
        plt.sca(ax[1,i])
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, zscore(reg,nan_policy='omit'), t_before_epoch = 0.2)
        opto = np.where(opto_block == 1)[0]
        reward = np.where(opto_block == 0)[0]
        incorrect = np.where(feedback == -1)[0]
        opto = np.intersect1d(opto, incorrect)
        reward = np.intersect1d(reward, incorrect)
        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = opto, t_range = [-0.2, 3.0])
        condition2 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = reward, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
            condition2 = (condition2[0]-F2, condition2[1])
        plot_psth(condition1, color='orange', plot_error=True)
        plot_psth(condition2, color='dodgerblue', plot_error=True)
        plt.title(reg_names[i])


# heatmap for every trial,
fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, NAcc, t_before_epoch = 0.5)
opto = np.where(opto_block== 1)[0]
reward = np.where(opto_block == 0)[0]
correct = np.where(feedback == 1)[0]       
opto = np.intersect1d(opto, correct)
reward = np.intersect1d(reward, correct)
condition1 = fp_psth(fluo_time_in_trials,
            fluo_in_trials,
            feedback_times, trial_list = opto, t_range = [-0.5, 3.0])
condition2 = fp_psth(fluo_time_in_trials,
            fluo_in_trials,
            feedback_times, trial_list = reward, t_range = [-0.5, 3.0])
lim1= min(np.nanmin(condition1[0]), np.nanmin(condition2[0]))
lim2= max(np.nanmax(condition1[0]), np.nanmax(condition2[0]))

xticks = condition1[1]
xticks = np.around(xticks, decimals=2)
fig,ax=plt.subplots(1,2, sharey=True)
plt.sca(ax[0])
sns.heatmap(condition1[0][opto], vmin=lim1, vmax=lim2, cmap='viridis')
plt.yticks(np.arange(0,len(opto),10),np.arange(0,len(opto),10))
plt.xticks(np.arange(len(xticks))[::10],xticks[::10], rotation=90)
plt.title('Opto Blocks')
plt.sca(ax[1])
sns.heatmap(condition2[0][reward], vmin=lim1, vmax=lim2,cmap='viridis')
plt.yticks(np.arange(0,len(reward),10),np.arange(0,len(reward),10))
plt.xticks(np.arange(len(xticks))[::10],xticks[::10], rotation=90)
plt.title('Reward Blocks')

