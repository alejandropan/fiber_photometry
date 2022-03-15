import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from numpy.random import default_rng
from scipy.signal import find_peaks

def se_compute_me(frames):
	'''
		Compute the mean motion energy per frame for a sequence of frames in place.
		Helps with memory efficiency when dealing with extremely large matrices. 
	'''

	mot_eng = np.zeros(frames.shape[0] - 1)
	for i in range(1, frames.shape[0]):
		abs_diff = np.abs(frames[i, :, :] - frames[i - 1, :, :])
		mot_eng[i - 1] = np.mean(abs_diff)

	return mot_eng

def compute_intensity(frames):
	return np.mean(frames, axis=(1, 2))

def get_ROI(filename, num_frames=500, start_frame=0):
	cap = cv2.VideoCapture(filename)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames = np.zeros((num_frames, height, width))

	i = 0
	frames_grabbed = 0
	ret = True
	while (i < num_frames + start_frame and ret):
		ret, img = cap.read()	
		if i >= start_frame:
			frames[frames_grabbed, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			frames_grabbed += 1

		i += 1
	
	cap.release()

	mean_img = np.mean(frames, axis=0) / 255 # CV2 multiplies float arrays by 255
	cv2.startWindowThread()
	x, y, dx, dy = cv2.selectROI(mean_img)
	cv2.destroyAllWindows()

	return (x, y, dx, dy)

def crop_ROI(filename, roi):
	''' 
		Given a bunch of frames, take their mean and prompt the user to pick an ROI with which to filter
	'''
	x, y, dx, dy = roi
	cap = cv2.VideoCapture(filename)

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	width = dx
	height = dy
	frames = np.zeros((total_frames, height, width))

	i = 0
	ret = True
	while (i < total_frames and ret):
		ret, frame = cap.read()
		frames[i, :, :] = cv2.cvtColor(frame[y : y + dy, x : x + dx], cv2.COLOR_BGR2GRAY)
		i += 1
	
	cap.release()

	return frames

def load_camera_timestamps(filename):
	''' Load timestamps in filename to list of strings. '''
	timestamps = []
	with open(filename) as f:
		for line in f:
			timestamps.append(line)

	return timestamps

def sliding_window(X, window_radius, mode='mean'):
    filtered_sig = np.zeros(len(X) - (2 * window_radius))
    for i in range(window_radius, len(X) - window_radius):
        if mode == 'mean':
            X0 = np.mean(X[i - window_radius : i + window_radius])
        elif mode == 'median': 
            X0 = np.median(X[i - window_radius : i + window_radius])
        
        filtered_sig[i - window_radius] = (X[i] - X0) / X0
    
    return np.array(filtered_sig)

def sig_normalize(X):
    return (X - np.nanmean(X)) / np.nanmean(X)

def dII(intensity, sr=30):
    mu = np.mean(intensity) 
    return (intensity - mu) / mu

def intensity_to_lick(intensity, thresh=0.75, ret_idxs=False):
	# Begin by normalizing the intensity
	norm_intensity = dII(intensity)

	# Search for peaks
	peak_idxs, props = find_peaks(norm_intensity, height=thresh)

	# Convert to lick array
	licks = np.zeros_like(intensity)
	licks.fill(np.nan)
	licks[peak_idxs] = props['peak_heights']

	if ret_idxs:
		return licks, peak_idxs
	else:
		return licks


# Testing script
if __name__ == '__main__':
	filename = './test_data/a_mouse_good/raw_video_data/_iblrig_leftCamera.raw.mp4'
	print('Getting ROI...', flush=True)
	roi = get_ROI(filename)
	print('Cropping ROI...', flush=True)
	cropped_frames = crop_ROI(filename, roi)
	print('Writing...', flush=True)
	me = se_compute_me(cropped_frames)

	# data = sio.ffprobe(filename)['video']
	# rate = data['@r_frame_rate']
	# T = np.int(data['@nb_frames'])

	# # Write vid
	# inputdict = {'-r': rate}
	# outputdict = {'-vcodec': 'libx264',
	# 			  '-r': rate}

	# writer = sio.FFmpegWriter('test_crop.mp4', inputdict=inputdict, outputdict=outputdict)

	# for i in range(cropped_frames.shape[0]):
	# 	print(i, flush=True)
	# 	writer.writeFrame(cropped_frames[i])

	# writer.close()