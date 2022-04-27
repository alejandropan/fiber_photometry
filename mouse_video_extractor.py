import sys
import os
from pathlib import Path
from ibllib.io.ffmpeg import iblrig_video_compression
from os import listdir
from os.path import isfile, join

topdir = sys.argv[1]
os.chdir(topdir)
all_subdirs = [x[0] for x in os.walk('./') if 'raw_video_data' in x[0]]

for sdx, sessiondir in enumerate(all_subdirs):
	if 'raw_video_data' not in sessiondir: # Skip all directories that don't end in raw_video_data
		continue

	files = [f for f in listdir(sessiondir) if isfile(join(sessiondir, f))]
	if '_iblrig_leftCamera.raw.avi' not in files: # Skip all directories where raw_video_data does not actually contain a video
		continue

	if '_iblrig_leftCamera.raw.mp4' in files: # Skip all directories that we have already compressed
		continue

	print('%d/%d %s' % (sdx, len(all_subdirs), sessiondir), end=' ', flush=True)
	session_path = Path(sessiondir[:-14])
	command = ('ffmpeg -i {file_in} -y -codec:v libx264 -preset slow -crf 29 '
    		   '-nostats -loglevel 0 -codec:a copy {file_out}')

	iblrig_video_compression(session_path, command)
	print('compressed')