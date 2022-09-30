from DAQ_extractor import *
import sys
import os
from pathlib import Path
from os import listdir
from os.path import isfile, join
from screen_extractor import*
from ibllib.io.extractors.training_trials import StimOffTriggerTimes


def extract_all_features(ses):
	#extract_all(ses, save=True, extra_classes=[StimOffTriggerTimes])
	#extract_all_wheel(ses,  save=True)
	extract_fp_daq(ses, save=True)
	session_labeler(ses)
	extract_screen_time(ses)
	
if __name__ == "__main__":
	topdir = sys.argv[1]
	os.chdir(topdir)
	all_subdirs = [x[0] for x in os.walk(topdir) if 'raw_video_data' in x[0]]
	errors = []
	for sdx, sessiondir in enumerate(all_subdirs):
		if 'raw_video_data' not in sessiondir:
			continue
		if 'raw_video_data/' in sessiondir:
			continue
		session_path = sessiondir[:-14]
		print(session_path)
		try:
			extract_all_features(session_path)
		except:
			errors.append(session_path)
	print(errors)



