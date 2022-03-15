# check if any of the session had pulses for both types of wavelengths

from pathlib import Path
import pandas as pd

animal  = Path('/Volumes/witten/Alex/Data/Subjects/fip_14')

sessions = []
pulses = []

for ses in animal.iterdir():
  print(ses)
  if  Path(ses.as_posix()+'/001/alf/fp_data/').is_dir():
    try:
      fp_data = pd.read_csv(ses.as_posix()+'/001/alf/fp_data/FPdata')
    except:
      fp_data = pd.read_csv(ses.as_posix()+'/001/alf/fp_data/FPdata.csv')

    pulses.append(fp_data.loc[fp_data['Flags']>=10,'Flags'].unique().shape[0])
    sessions.append(ses)
