from signal_summary import *
from pathlib import Path
import os

if __name__ == "__main__":
    DATA_FOLDER = '/Volumes/witten/Alex/Data/Subjects/'
    mice = ['fip_24']
    for mouse in mice:
        mouse_folder = Path(DATA_FOLDER+mouse)
        list_subfolders_with_paths = [f.path for f in os.scandir(mouse_folder) if f.is_dir()] 
        list_subfolders_with_paths.sort()
        for fld in list_subfolders_with_paths:
            try:
                session_labeler(fld+'/001')
                print(fld)
            except:
                continue
