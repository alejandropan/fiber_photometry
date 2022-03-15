import numpy as np
import matplotlib.pyplot as plt

ses = '/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-13/001'
_, ax = plt.subplots(1,3)
plt.sca(ax[0])
spikes = np.load(ses + '/alf/probe00/pykilosort/spikes.times.npy')
selection =np.load(ses + '/alf/probe00/pykilosort/clusters_selection.npy')
depths = np.load(ses+ '/alf/probe00/pykilosort/spikes.depths.npy')
spikes_clusters = np.load(ses+'/alf/probe00/pykilosort/spikes.clusters.npy')
spikes_selection = spikes[np.where(np.isin(spikes_clusters,selection))]
spikes_selection = spikes_selection[np.where((spikes_selection<1060)&(spikes_selection>1000))]
depths_selection = depths[np.where(np.isin(spikes_clusters,selection))]
depths_selection = depths_selection[np.where((spikes_selection<1060)&(spikes_selection>1000))]
plt.scatter(spikes_selection,depths_selection, s=1, color='k')

plt.sca(ax[1])
spikes = np.load(ses + '/alf/probe01/pykilosort/spikes.times.npy')
selection =np.load(ses + '/alf/probe01/pykilosort/clusters_selection.npy')
depths = np.load(ses+ '/alf/probe01/pykilosort/spikes.depths.npy')
spikes_clusters = np.load(ses+'/alf/probe01/pykilosort/spikes.clusters.npy')
spikes_selection = spikes[np.where(np.isin(spikes_clusters,selection))]
spikes_selection = spikes_selection[np.where((spikes_selection<1060)&(spikes_selection>1000))]
depths_selection = depths[np.where(np.isin(spikes_clusters,selection))]
depths_selection = depths_selection[np.where((spikes_selection<1060)&(spikes_selection>1000))]
plt.scatter(spikes_selection,depths_selection, s=1, color='k')

plt.sca(ax[2])
spikes = np.load(ses + '/alf/probe02/pykilosort/spikes.times.npy')
selection =np.load(ses + '/alf/probe02/pykilosort/clusters_selection.npy')
depths = np.load(ses+ '/alf/probe02/pykilosort/spikes.depths.npy')
spikes_clusters = np.load(ses+'/alf/probe02/pykilosort/spikes.clusters.npy')
spikes_selection = spikes[np.where(np.isin(spikes_clusters,selection))]
spikes_selection = spikes_selection[np.where((spikes_selection<1060)&(spikes_selection>1000))]
depths_selection = depths[np.where(np.isin(spikes_clusters,selection))]
depths_selection = depths_selection[np.where((spikes_selection<1060)&(spikes_selection>1000))]
plt.scatter(spikes_selection,depths_selection, s=1, color='k')

#######

loc = np.load(ses+'/alf/probe00/pykilosort/channels.locations.npy', allow_pickle=True)
loc = loc.astype('str')
for l in np.unique(loc):
    plt.hlines((np.where(loc==l)[0][0]+1)*10,0,1)
    plt.text(0, (np.where(loc==l)[0][0]+1)*10, l)


loc = np.load(ses+'/alf/probe01/pykilosort/channels.locations.npy', allow_pickle=True)
loc = loc.astype('str')
for l in np.unique(loc):
    plt.hlines((np.where(loc==l)[0][0]+1)*10,0,1)
    plt.text(0, (np.where(loc==l)[0][0]+1)*10, l)

loc = np.load(ses+'/alf/probe02/pykilosort/channels.locations.npy', allow_pickle=True)
loc = loc.astype('str')
for l in np.unique(loc):
    plt.hlines((np.where(loc==l)[0][0]+1)*10,0,1)
    plt.text(0, (np.where(loc==l)[0][0]+1)*10, l)

