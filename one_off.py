import os
import numpy as np
import matplotlib.pyplot as plt


root = '/Volumes/witten/Alex/Data/Subjects/fip_14/'
ses = os.listdir(root)

for i, s in enumerate(ses):
  try:
    extract_fp_daq(root + s +'/001', save=True)
  except:
    print(s)
