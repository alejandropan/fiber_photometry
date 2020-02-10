#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:04:18 2020

@author: alex
"""

import cv2
import numpy as np
import ssv
import dateutil.parser
from time import mktime
from path import Path
import sys

def ssv_2_array(ssv_file, video_file, mode='fiber'):
    '''
    Extract ssv timestamps files and transforms it 
    into a numpy array. Note: time in unix time for easier handling

    Parameters
    ----------

    ssv_file : ssv file with fluorescence and camera times.
    video_file : mp4 or avi files from camera or fiber.
    mode : whether it has fluorescence signal or not, 'fiber' does
    'left' does not, optional. The default is 'fiber'.
    
    Returns 
    -------
    numpy array with camera timestamps in
    '''
    
    cap = cv2.VideoCapture(video_file)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ssv_file = ssv.loadf(ssv_file)
    if mode == 'fiber':
        y = np.empty([int(nframes),4])
        for n,i in enumerate(ssv_file[0][0].split('\n')[:-1]): 
            j = i.split(" ")
            r = j[0].split(",")
            date = dateutil.parser.parse(r[1])
            # Convert date to unix date
            y[n,0] = r[0]
            y[n,1] = mktime(date.timetuple())*1e3 + date.microsecond/1e3
            y[n,2:] = r[2:]
        
    elif mode == 'left':
        y = np.empty([int(nframes),3])
        for n,i in enumerate(ssv_file[0][0].split('\n')[:-1]): 
            j = i.split(" ")
            date = dateutil.parser.parse(j[1])
            y[n,0] =  j[0]
            y[n,1] = mktime(date.timetuple())*1e3 + date.microsecond/1e3
            y[n,2] = np.nan
            
    return y


def extract_fp_time(session_path):
    '''
    Obtain bpod times for fp signals
    
    Parameters
    ----------
    session_path : string with session path

    Returns
    -------
    fp_bpodtime : times of fiber frames in bpod time

    '''
    # Get paths
    ssv_fiber = session_path + '/raw_video_data/' + '_iblrig_fiberCamera.timestamps.ssv'
    ssv_camera = session_path + '/raw_video_data/' + '_iblrig_leftCamera.timestamps.ssv'
    fiber_video = session_path + '/raw_video_data/' + '_iblrig_fiberCamera.raw.mp4'
    left_video = session_path + '/raw_video_data/' + '_iblrig_leftCamera.raw.mp4'
    camera_bpod = session_path + '/alf/' + '_ibl_leftCamera.times.npy'
    
    #First get path to different files
    
    fiber = ssv_2_array(ssv_fiber, fiber_video)[:,1]
    camera = ssv_2_array(ssv_camera, left_video, mode='left')[:,1]
    camera_bpod = np.load(camera_bpod)   
    # First calculate fiber times in relation to left camera time
    fp_leftcamtime = fiber - camera[0]
    # msec to sec
    fp_leftcamtime = fp_leftcamtime/1000
    # Then calibrate to bpod time
    fp_bpodtime = fp_leftcamtime + camera_bpod[0]
    
    # Save times
    fpath = Path(session_path).joinpath('alf', '_ibl_fluo.times.npy') 
    np.save(fpath, fp_bpodtime)
    
    return fp_bpodtime

def extract_fp_fluorescence(session_path):
    '''
    Extract fluorescence values for each fiber
    
    Parameters
    ----------
    session_path : string with session path


    Returns
    -------
    avg_noise : average noise fluorescence per frame from ROI outside
    path cable
    avg_loc1 : average fluorescence per frame for roi 1
    avg_loc2 : average fluorescence per frame for roi 2
    avg_loc3 : average fluorescence per frame for roi 3
    avg_loc4 : average fluorescence per frame for roi 4

    '''
    
    # Get paths
    ssv_loc1 = session_path + '/raw_video_data/' + '_iblrig_loc1Camera.timestamps.ssv'
    ssv_loc2 = session_path + '/raw_video_data/' + '_iblrig_loc2Camera.timestamps.ssv'
    ssv_loc3 = session_path + '/raw_video_data/' + '_iblrig_loc3Camera.timestamps.ssv'
    ssv_loc4 = session_path + '/raw_video_data/' + '_iblrig_loc4Camera.timestamps.ssv'
    ssv_fiber = session_path + '/raw_video_data/' + '_iblrig_fiberCamera.timestamps.ssv'
    fiber_video = session_path + '/raw_video_data/' + '_iblrig_fiberCamera.raw.mp4'
    
    # Extract noise roi from fiber timestamps file
    fiber = ssv_2_array(ssv_fiber, fiber_video)
    
    # Extract fluorescence
    loc1 = ssv_2_array(ssv_loc1, fiber_video)
    loc2 = ssv_2_array(ssv_loc2, fiber_video)
    loc3 = ssv_2_array(ssv_loc3, fiber_video)
    loc4 = ssv_2_array(ssv_loc4, fiber_video)
    
    # Measure noise
    avg_noise =  fiber[:,2]
    
    # Subtract noise 
    avg_loc1 = loc1[:,3] - avg_noise
    avg_loc2 = loc2[:,3] - avg_noise
    avg_loc3 = loc3[:,3] - avg_noise
    avg_loc4 = loc4[:,3] - avg_noise
    
    # Save everything
    fpath = Path(session_path).joinpath('alf', '_ibl_noise.fluo.npy') 
    np.save(fpath, avg_noise)
    fpath = Path(session_path).joinpath('alf', '_ibl_loc1.fluo.npy') 
    np.save(fpath, avg_loc1)
    fpath = Path(session_path).joinpath('alf', '_ibl_loc2.fluo.npy') 
    np.save(fpath, avg_loc2)
    fpath = Path(session_path).joinpath('alf', '_ibl_loc3.fluo.npy') 
    np.save(fpath, avg_loc3)
    fpath = Path(session_path).joinpath('alf', '_ibl_loc4.fluo.npy') 
    np.save(fpath, avg_loc4)
    
    return avg_noise, avg_loc1, avg_loc2, avg_loc3, avg_loc4


if __name__ == "__main__":
    session_path = sys.argv[1]
    extract_fp_time(session_path)
    extract_fp_fluorescence(session_path)

        





