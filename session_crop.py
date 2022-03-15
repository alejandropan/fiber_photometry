from code import interact
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pandas as pd


def run_gui(data, prev_reg_selection=None, title=None, facecolor='black', palette=None):
    selection = []
    previous_selection_len=0
    while True:
        fig, ax = plt.subplots()
        ax.plot(data)
        plt.title(title)
        if prev_reg_selection is not None:
            for i in np.arange(len(prev_reg_selection)):
                for j in np.arange(len(prev_reg_selection[i])):
                    ax.axvspan(prev_reg_selection[i][j][0], prev_reg_selection[i][j][1], color=palette[i], alpha=.25)
        for i in np.arange(len(selection)):
            ax.axvspan(selection[i][0], selection[i][1], color=facecolor, alpha=.25)
        # create the gray span to show what we've selected
        selected = ax.axvspan(0, 0, color='black', alpha=.25)
        selected.set_visible(False)

        def onselect(xmin, xmax):
            x0=int(xmin)
            x1=int(xmax)
            ax.set_title(f"The selected range goes from {x0} from {x1}")
            # update the visualization of the selected area
            selected.set_visible(True)
            xy = selected.get_xy()
            xy[[0,1],0] = xmin
            xy[[2,3],0] = xmax
            selected.set_xy(xy)
            fig.canvas.draw()

        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, span_stays=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))
        plt.show()
        if selected.xy[0,0]!=selected.xy[2,0]: # This effectively checks if something was selected
            selection.append([selected.xy[0,0], selected.xy[2,0]])
        if len(selection)==previous_selection_len:
            break
        previous_selection_len=len(selection)
    return selection

def run_all_regions(data):
    # Data is the FPdata dataframe
    titles = ['DMS', 'NAcc', 'DLS']
    regions =  ['DMS', 'NAcc', 'DLS']
    pal = ['dodgerblue', 'orange', 'magenta']
    selection1=[]
    col_selections = []
    for i, reg in enumerate(regions):
        selection1 = run_gui(data[reg], prev_reg_selection=col_selections, 
                        title=titles[i], facecolor=pal[i], palette=pal)
        col_selections.append(selection1)
    return col_selections

