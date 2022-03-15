import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def raw_to_dff(gcamp465, gcamp405, sp = 25, cutoff=2):
    '''
    Al la lerner,deisseroth bbleach correction
    GCaMP465: gcamp signal
    GCaMP405: isosbestic signal
    '''
    b,a = butter(4, cutoff/(sp/2),'lowpass') # butter filter: order 4, 2hz objective frequency, lowpass
    gcamp465_filtered = filtfilt(b,a,gcamp465)
    gcamp405_filtered = filtfilt(b,a,gcamp405)
    p = np.polyfit(gcamp405,gcamp465,1)
    gcamp405_fitted = p[0]*gcamp405_filtered+p[1]
    dff =  gcamp465_filtered/gcamp405_fitted
    return dff