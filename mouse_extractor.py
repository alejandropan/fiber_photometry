from DAQ_extractor import *
import sys
import os
from pathlib import Path
from os import listdir
from os.path import isfile, join

def extract_all_features(ses):
	extract_all(ses, save=True)
	extract_all_wheel(ses,  save=True)
	extract_fp_daq(ses, save=True)
	session_labeler(ses)
	
if __name__ == "__main__":
	topdir = sys.argv[1]
	os.chdir(topdir)
	all_subdirs = [x[0] for x in os.walk(topdir) if 'raw_fp_data' in x[0]]
	errors = []
	for sdx, sessiondir in enumerate(all_subdirs):
		if 'raw_fp_data' not in sessiondir:
			continue
		if 'raw_fp_data/' in sessiondir:
			continue
		session_path = sessiondir[:-12]
		print(session_path)
		try:
			extract_all_features(session_path)
		except:
			errors.append(session_path)
	print(errors)
