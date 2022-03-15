'''
Syncing test
415 Region 0 facing an LED sending pulses to DAQ
415 laser send TTLs to DAQ
100Hz, 1-415,2-470, 3-None, 4-None
Used 415 first because the filter led red LED light through
'''
from nptdms import TdmsFile
import pandas as pd

ses='/Users/alex/Downloads/first_test'
fp_data = pd.read_csv(ses + '/FP415')

for file in os.listdir(ses):
    if file.endswith(".tdms"):
        td_f = file
tdms_file = TdmsFile.read(ses + '/'+ td_f)
signal =pd.DataFrame()
signal['DAQ_FP'] = tdms_file._channel_data["/'Analog'/'AI0'"].data
signal['DAQ_light'] = tdms_file._channel_data["/'Analog'/'AI7'"].data
signal['DAQ_FP'] = 1 * (signal['DAQ_FP']>=4)
signal['DAQ_light'] = 1 * (signal['DAQ_light']>=0.1)

signal.loc[np.where(signal['DAQ_FP'].diff()==1)[0], 'TTL_change'] = 1
signal.loc[np.where(signal['DAQ_light'].diff()==1)[0], 'TTL_change'] = 1
sample_ITI  = np.median(np.diff(signal.loc[signal['TTL_change']==1].index))

print('DAQ recived '+ len(np.where(signal['DAQ_FP'].diff()==1)[0]) - len(fp_data)
        + ' more pulses than it saved')

fp_data['from_light'] = 1 * (fp_data['Region0R']>=0.1)
fp_data['from_DAQ_light'] = np.nan

fp_data.iloc(np.where(fp_data['from_light'].diff()==1)[0],
            fp_data.columns.get_loc('from_DAQ_light')) = \
            np.where(signal['DAQ_light'].diff()==1)[0]

# Align events
fp_data['DAQ_timestamp'] = np.nan
daq_idx = fp_data.columns.get_loc('DAQ_timestamp')
fp_data.iloc[:,daq_idx] = \
np.where(signal['DAQ_FP'].diff()==1)[0][:len(fp_data)]

#####


fp_data['DAQ_timestamp_del_end'] = np.nan
daq_idx = fp_data.columns.get_loc('DAQ_timestamp_del_end')
fp_data.iloc[:,daq_idx] = \
np.where(signal['DAQ_FP'].diff()==1)[0][:len(fp_data)]

fp_data['DAQ_timestamp_del_start'] = np.nan
daq_idx = fp_data.columns.get_loc('DAQ_timestamp_del_start')
fp_data.iloc[:,daq_idx] = \
np.where(signal['DAQ_FP'].diff()==1)[0][len(fp_data):]
