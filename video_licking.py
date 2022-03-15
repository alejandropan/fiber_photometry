import numpy as np
import matplotlib.pyplot as plt
import analyse_video as av
from psths import *
import pims
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
import analyse_video as av
sns.set(context="poster")

# Settings
LOAD_PREVIOUS=True
MAKE_VIDEO=False
PLOT=True
# session
session = '/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-09/001'
video_filename = session + '/raw_video_data/_iblrig_leftCamera.raw.mp4'
# Load video
camera_times = np.load(session + '/alf/_ibl_leftCamera.times.npy')
crop_type = 'tongue_crop'
crop_file = crop_type + '.npz'


if LOAD_PREVIOUS==False:
    # Raw data
    reset_sessions = []
    num_frames = 500
    start_frame = 0
    print(session)
    print('\tGetting ROI...', end=' ', flush=True)
    roi = av.get_ROI(video_filename, num_frames=num_frames, start_frame=start_frame)
    print('Done', flush=True)

    print('\tSaving...', end=' ', flush=True)
    if roi == (0, 0, 0, 0):
        np.savez(session + '/raw_video_data/' + crop_file, roi=None)
    else:
        np.savez(session + '/raw_video_data/' + crop_file, roi=roi)
    print('Done', flush=True)
    # Measure number of licks 
    crop_type = 'tongue_crop'
    crop_file = crop_type + '.npz'
    lick_thresh = 0.5
    licks = []
    camera_timestamps = []
    # Load ROI
    # Crop frames
    print('\tCropping...', end=' ', flush=True)
    cropped_frames = av.crop_ROI(video_filename, roi)
    print('Done.')

    # Get intensity
    intensity = av.compute_intensity(cropped_frames)
    lick = av.intensity_to_lick(intensity, thresh=lick_thresh)
    np.save(session + '/raw_video_data/intensity.npy', intensity)
    np.save(session + '/raw_video_data/licks.npy', lick)

else:
    roi = np.load(session + '/raw_video_data/' + crop_file)['roi']
    intensity = np.load(session + '/raw_video_data/intensity.npy')
    lick = np.load(session + '/raw_video_data/licks.npy')
    lick_binary = 1*(lick>0)

# Make video
if MAKE_VIDEO==True:
    def update_video(j, iline, lline, im, video, intensities, licks, times, window_size_frames):
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        if j % 1000 == 0:
            print('%d/%d' % (j, frames_to_animate))
        start_frame = j
        end_frame = j + window_size_frames

        im.set_data(video[j])
        iline.set_data(times, intensities[start_frame : end_frame])
        lline.set_data(times, licks[start_frame : end_frame])

        return (iline, lline)

    POST = 5
    PRE = 2
    fr = 30
    window_size_frames = (PRE + POST) * fr

    mins = 10
    frames_to_animate = int(mins * 60 * fr)
    lick_thresh = 0.50
    video = pims.Video(video_filename)
    roi_fn = session + '/raw_video_data/' + crop_file
    # Load licking data
    print('\tLoading data')
    intensities = av.sig_normalize(intensity)
    intensities = np.concatenate((np.zeros(PRE * fr), intensities))
    intensities[:PRE * fr] = np.nan
    licks = np.concatenate((np.zeros(PRE * fr), lick))
    licks[:PRE * fr] = np.nan

    ## Plot
    print('\tPlotting first frame')
    # Plot params
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20,6),tight_layout=True)

    gs = gridspec.GridSpec(1, 2)
    imax = fig.add_subplot(gs[0,0])
    traceax = fig.add_subplot(gs[0,1])

    # Plot first frame of video
    im = imax.imshow(video.get_frame(0))
    imax.axis('off')

    # Load ROI
    x, y, dx, dy = np.load(roi_fn)['roi']
    rect = patches.Rectangle((x, y), dx, dy, linewidth=1, edgecolor='r', facecolor='none')
    imax.add_patch(rect)

    # Plot first frame of trace
    times = np.linspace(-PRE, POST, window_size_frames)
    intense_line = traceax.plot(times, intensities[:window_size_frames], c='dodgerblue')[0]
    lick_line = traceax.plot(times, licks[:window_size_frames], 'o', c='yellow')[0]
    traceax.set_ylim(-0.5, 4)
    traceax.set_xlim(-PRE, POST)
    vline = traceax.axvline(0, linestyle='--', color='white')
        
    traceax.set_ylabel('mean ROI intensity')
    traceax.set_xticklabels([])
    sns.despine()
    traceax.grid(False)

    # Animate remaining frames
    print('\tAnimating remaining frames')
    anim = FuncAnimation(fig, update_video,
                            frames=frames_to_animate,
                            interval= 1/fr * 1000,
                            fargs=(intense_line, lick_line, im, video, intensities, 
                            licks, times, window_size_frames))

    anim.save(session + '/raw_video_data/lick_detection.mp4', dpi=96)

if PLOT==True:
    #Load alf info
    camera_times = np.load(session + '/alf/_ibl_leftCamera.times.npy')
    opto_block = np.load (session +'/alf/_ibl_trials.opto_block.npy')
    feedback_time = np.load (session +'/alf/_ibl_trials.feedback_times.npy')
    feedback = np.load(session + '/alf/_ibl_trials.feedbackType.npy')
    trial_start = np.load(session + '/alf/_ibl_trials.goCueTrigger_times.npy')
    licks_in_trials, lick_time_in_trials = divide_in_trials(camera_times, trial_start, lick_binary)

    opto = np.where(opto_block== 1)[0]
    reward = np.where(opto_block == 0)[0]
    correct = np.where(feedback == 1)[0]       
    incorrect = np.where(feedback == -1)[0]       
    ropto = np.intersect1d(opto, correct)
    rreward = np.intersect1d(reward, correct)
    eopto = np.intersect1d(opto, incorrect)
    ereward = np.intersect1d(reward, incorrect)


    # Plot average
    lick_psth_laser = fp_psth(lick_time_in_trials, licks_in_trials, feedback_time[:-1], trial_list=ropto[:-1], t_range = [-0.5, 3.0], T_BIN = 0.035)
    plot_psth(lick_psth_laser, color='orange')
    lick_psth_water = fp_psth(lick_time_in_trials, licks_in_trials, feedback_time[:-1], trial_list=rreward[:-1], t_range = [-0.5, 3.0], T_BIN = 0.035)
    plot_psth(lick_psth_water, color='dodgerblue')

    # heatmap for every trial, feedback
    condition1 = fp_psth(lick_time_in_trials,
                licks_in_trials,
                feedback_time[:-1], trial_list = ropto[:-1], t_range = [-0.5, 3.0], T_BIN = 0.035)
    condition2 = fp_psth(lick_time_in_trials,
                licks_in_trials,
                feedback_time[:-1], trial_list = rreward[:-1], t_range = [-0.5, 3.0], T_BIN = 0.035)

    lim1= min(np.nanmin(condition1[0]), np.nanmin(condition2[0]))
    lim2= max(np.nanmax(condition1[0]), np.nanmax(condition2[0]))
    xticks = condition1[1]
    xticks = np.around(xticks, decimals=2)
    fig,ax=plt.subplots(1,2, sharey=True)
    plt.sca(ax[0])
    sns.heatmap(condition1[0][ropto[:-1]], vmin=lim1, vmax=lim2, cmap='binary')
    plt.vlines(np.abs(condition1[1] - 0).argmin(),0,len(condition1[0][ropto[:-1]]), color='r', linestyles='dashed')
    plt.yticks(np.arange(0,len(ropto[:-1]),10),np.arange(0,len(ropto[:-1]),10))
    plt.xticks(np.arange(len(xticks))[::10],xticks[::10], rotation=90)
    plt.title('Opto Blocks')
    plt.sca(ax[1])
    sns.heatmap(condition2[0][rreward[:-1]], vmin=lim1, vmax=lim2,cmap='binary')
    plt.vlines(np.abs(condition2[1] - 0).argmin(),0,len(condition1[0][rreward[:-1]]), color='r', linestyles='dashed')
    plt.yticks(np.arange(0,len(rreward[:-1]),10),np.arange(0,len(rreward[:-1]),10))
    plt.xticks(np.arange(len(xticks))[::10],xticks[::10], rotation=90)
    plt.title('Reward Blocks')

    #Compare water to psth of signal
    ses = session

    fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
    NAcc = np.load(ses +'/alf/_ibl_trials.NAcc.npy')
    DLS = np.load(ses +'/alf/_ibl_trials.DLS.npy')
    DMS = np.load(ses +'/alf/_ibl_trials.DMS.npy')
    cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
    opto_block = np.load (ses +'/alf/_ibl_trials.opto_block.npy')
    response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
    left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
    right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
    l_trials = np.nan_to_num(left_trials)
    r_trials = np.nan_to_num(right_trials)
    signed_contrast = r_trials - l_trials
    feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
    choice = np.load(ses + '/alf/_ibl_trials.choice.npy')[:-1]
    opto_block = opto_block[:-1]
    response_times = response_times[:-1]
    l_trials = l_trials[:-1]
    r_trials = r_trials[:-1]
    signed_contrast = signed_contrast[:-1]
    feedback = feedback[:-1]
    stimOn_times  = np.load(ses + '/alf/_ibl_trials.stimOn_times.npy')


    lick_w_psth = fp_psth(lick_time_in_trials, licks_in_trials, feedback_time[:-1], trial_list=rreward[:-1], t_range = [-0.5, 3.0], T_BIN = 0.035)
    fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, NAcc)
    nacc_w_psth = fp_psth(fluo_time_in_trials, fluo_in_trials, feedback_time[:-1], trial_list = rreward[:-1], t_range = [-0.5, 3.0], T_BIN = 0.035)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.sca(ax2)
    plot_psth(lick_w_psth, color='k')
    plt.sca(ax1)
    plot_psth(nacc_w_psth, color='dodgerblue')
    plt.xlabel('Time from Feedback (s)')
    ax2.set_ylabel('Number Licks', color='k')
    ax1.set_ylabel('DF/F', color='dodgerblue')



