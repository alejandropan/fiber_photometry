from pathlib import Path
import os

if __name__ == "__main__":
    DATA_FOLDER = '/Volumes/witten/Alex/Data/Subjects/'
    mice = ['fip_23','fip_25']
    for mouse in mice:
        mouse_folder = Path(DATA_FOLDER+mouse)
        list_subfolders_with_paths = [f.path for f in os.scandir(mouse_folder) if f.is_dir()] 
        list_subfolders_with_paths.sort()
        for fld in list_subfolders_with_paths:
            try:
                path = fld+'/001'
                os.rename(path+'/alf/_ibl_trials.DMS.npy', path+'/alf/_ibl_trials.DLS_temp.npy')
                os.rename(path+'/alf/_ibl_trials.DLS.npy', path+'/alf/_ibl_trials.DMS.npy')
                os.rename(path+'/alf/_ibl_trials.DLS_temp.npy', path+'/alf/_ibl_trials.DLS.npy')
                print(str(path))
            except:
                print('Error in '+ str(path))
    