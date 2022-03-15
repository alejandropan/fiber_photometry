import numpy as np
import pandas as pd

dates = \
 [
'2021-03-08',
'2021-03-09',
'2021-03-10',
'2021-03-11',
'2021-03-12',
]

fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data = pd.read_csv('/Volumes/witten/Alex/Data/Subjects/fip_13/'+day+'/001/alf/fp_data/FP470')
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='Region2G', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('Raw Fluorescence')
plt.title('5 consecutive day Fip13 - Raw')
plt.xlim(10000,10750)
plt.sca(ax[1])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data=pd.DataFrame()
    NAcc = np.load('/Volumes/witten/Alex/Data/Subjects/fip_13/'+day+'/001/alf/_ibl_trials.DLS.npy')
    data['NAcc'] = NAcc
    data['NAcc'] = NAcc
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='NAcc', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('DF/F')
plt.title('5 consecutive day Fip13 - DF/F')
plt.xlim(10000,10750)




dates = \
[
'2021-04-12',
'2021-04-13',
'2021-04-14',
'2021-04-15',
'2021-04-16',
]

fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data = pd.read_csv('/Volumes/witten/Alex/Data/Subjects/fip_13/'+day+'/001/alf/fp_data/FP470')
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='Region2G', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('Raw Fluorescence')
plt.title('Last 5 consecutive day Fip13 - Raw')
plt.xlim(10000,10750)
plt.sca(ax[1])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data=pd.DataFrame()
    NAcc = np.load('/Volumes/witten/Alex/Data/Subjects/fip_13/'+day+'/001/alf/_ibl_trials.DLS.npy')
    data['NAcc'] = NAcc
    data['NAcc'] = NAcc
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='NAcc', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('DF/F')
plt.title('Last 5 consecutive day Fip13 - DF/F')
plt.xlim(10000,10750)






dates = \
 [
'2021-03-08',
'2021-03-09',
'2021-03-10',
'2021-03-11',
'2021-03-12',
]

fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data = pd.read_csv('/Volumes/witten/Alex/Data/Subjects/fip_13/'+day+'/001/alf/fp_data/FP470')
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='Region2G', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('Raw Fluorescence')
plt.title('5 consecutive day Fip14 - Raw')
plt.xlim(10000,10750)
plt.sca(ax[1])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data=pd.DataFrame()
    NAcc = np.load('/Volumes/witten/Alex/Data/Subjects/fip_13/'+day+'/001/alf/_ibl_trials.DLS.npy')
    data['NAcc'] = NAcc
    data['NAcc'] = NAcc
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='NAcc', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('DF/F')
plt.title('5 consecutive day Fip14 - DF/F')
plt.xlim(10000,10750)




dates = \
[
'2021-04-12',
'2021-04-13',
'2021-04-14',
'2021-04-15',
'2021-04-16',
]

fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data = pd.read_csv('/Volumes/witten/Alex/Data/Subjects/fip_14/'+day+'/001/alf/fp_data/FP470')
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='Region2G', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('Raw Fluorescence')
plt.title('Last 5 consecutive day Fip14 - Raw')
plt.xlim(10000,10750)
plt.sca(ax[1])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data=pd.DataFrame()
    NAcc = np.load('/Volumes/witten/Alex/Data/Subjects/fip_14/'+day+'/001/alf/_ibl_trials.DLS.npy')
    data['NAcc'] = NAcc
    data['NAcc'] = NAcc
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='NAcc', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('DF/F')
plt.title('Last 5 consecutive day Fip14 - DF/F')
plt.xlim(10000,10750)



dates = \
[
'2021-12-13',
'2021-12-14',
'2021-12-15',
'2021-12-16',
'2021-12-17']

fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data = pd.read_csv('/Volumes/witten/Alex/Data/Subjects/fip_24/'+day+'/001/alf/fp_data/FP470')
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='Region2G', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('Raw Fluorescence')
plt.title('5 consecutive day Fip25 - Raw')
plt.xlim(10000,10750)
plt.sca(ax[1])
ses_data = pd.DataFrame()
for i, day in enumerate(dates):
    data=pd.DataFrame()
    NAcc = np.load('/Volumes/witten/Alex/Data/Subjects/fip_24/'+day+'/001/alf/_ibl_trials.DLS.npy')
    data['NAcc'] = NAcc
    data['NAcc'] = NAcc
    data['day'] = i
    ses_data = pd.concat([ses_data,data.reset_index()])
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
sns.lineplot(data=ses_data.reset_index(),x='index', y='NAcc', hue='day', ci=None, palette='viridis')
plt.xlabel('Frames')
plt.ylabel('DF/F')
plt.title('5 consecutive day Fip25 - DF/F')
plt.xlim(10000,10750)
