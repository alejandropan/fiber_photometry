#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:47:45 2020

@author: alex
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)


fig, ax = plt.subplots(figsize=(10,6))
plt.xlim(0, 2000)
plt.ylim(13, np.max(raw_fluo))
plt.xlabel('Frame',fontsize=20)
plt.ylabel('F',fontsize=20)
plt.title('NaCC',fontsize=20)
ax.set_facecolor( (1, 1, 1))

t = 1800

%matplotlib qt 
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=t, repeat=True)
plt.show()
ani.save('loc2.mp4', writer=writer)

def animate(i):
    raw_fluopd = pd.DataFrame(data=raw_fluo, columns =['fluo'])
    raw_fluopd = raw_fluopd.iloc[:(i+1)] #select data range
    p = sns.lineplot(x=raw_fluopd.index, y='fluo', data=raw_fluopd, color="r")
    p.tick_params(labelsize=17)
    plt.setp(p.lines,linewidth=1)