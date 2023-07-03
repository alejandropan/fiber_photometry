# check autocorrelation
from scipy.stats import zscore

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]



def plot_autocorr (dms,dls):
    dl = autocorr(zscore(dls[int(dls.shape[0]/2):int(dls.shape[0]/2)+10000]))
    dm = autocorr(zscore(dms[int(dms.shape[0]/2):int(dms.shape[0]/2)+10000]))
    plt.plot(dl[:25])
    plt.plot(dm[:25])


sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_26/2022-04-07/001',
'/Volumes/witten/Alex/Data/Subjects/fip_27/2022-04-07/004/001',
'/Volumes/witten/Alex/Data/Subjects/fip_28/2022-04-06/001',
'/Volumes/witten/Alex/Data/Subjects/fip_29/2022-04-04/001',
'/Volumes/witten/Alex/Data/Subjects/fip_30/wheel_fixed',
'/Volumes/witten/Alex/Data/Subjects/fip_31/2022-06-10/001',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-03/001',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-05-20/001',
'/Volumes/witten/Alex/Data/Subjects/fip_34/2022-12-13/001',
'/Volumes/witten/Alex/Data/Subjects/fip_35/2022-12-23/001',
'/Volumes/witten/Alex/Data/Subjects/fip_36/2022-12-23/001',
'/Volumes/witten/Alex/Data/Subjects/fip_37/2022-12-22/001',
'/Volumes/witten/Alex/Data/Subjects/fip_38/2022-12-23/001'
]

fig, ax = plt.subplots(13)
for i, ses in enumerate(sessions):
    print(i)
    dls = np.load(ses + '/alf/_ibl_trials.DLS.npy')
    dms = np.load(ses + '/alf/_ibl_trials.DMS.npy')
    plt.sca(ax[i])
    plot_autocorr (dms,dls)



sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-06-13/001',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-06-14/001',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-06-15/001',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-06-16/001',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-06-22/001',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-06-10/002',
'/Volumes/witten/Alex/Data/Subjects/fip_33/2022-06-09/001'
]

fig, ax = plt.subplots(len(sessions))
for i, ses in enumerate(sessions):
    print(i)
    dls = np.load(ses + '/alf/_ibl_trials.DLS.npy')
    dms = np.load(ses + '/alf/_ibl_trials.DMS.npy')
    plt.sca(ax[i])
    plot_autocorr (dms,dls)


sessions = [
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-13/001',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-14/001',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-15/002',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-16/001',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-22/001',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-10/001',
'/Volumes/witten/Alex/Data/Subjects/fip_32/2022-06-09/001'
]

fig, ax = plt.subplots(len(sessions))
for i, ses in enumerate(sessions):
    print(i)
    dls = np.load(ses + '/alf/_ibl_trials.DLS.npy')
    dms = np.load(ses + '/alf/_ibl_trials.DMS.npy')
    plt.sca(ax[i])
    plot_autocorr (dms,dls)
