import sys
import os
from pathlib import Path
from ibllib.io.extractors.camera import extract_all
from os import listdir
from os.path import isfile, join

topdir = sys.argv[1]
os.chdir(topdir)
all_subdirs = [x[0] for x in os.walk('./')]

for sdx, sessiondir in enumerate(all_subdirs):
	if 'raw_video_data' not in sessiondir: # Skip all directories that don't end in raw_video_data
		continue

	files = [f for f in listdir(sessiondir) if isfile(join(sessiondir, f))]
	if '_iblrig_leftCamera.raw.mp4' not in files: # Skip all directories that we haven't already compressed
		continue


	# Delete GPIO file, if necessary
	if '_iblrig_leftCamera.GPIO.bin' in files:
		os.remove(join(sessiondir, '_iblrig_leftCamera.GPIO.bin'))

	print('%d/%d %s' % (sdx, len(all_subdirs), sessiondir), end=' ', flush=True)
	session_path = Path(sessiondir[:-14])
	try:
		extract_all(session_path, session_type='training')
		print('extracted')
	except:
		print('failed')
	