# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:38:43 2022

@author: DELL
"""
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt

from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
}
rcParams.update(config)


WHU_ious = [87.28,89.45,89.39,89.86,91.13]
Massachusetts_ious = [67.35,72.61,71.87,73.45,74.13]
Inria_ious = [78.43,79.25,78.49,80.48,81.05]

x = [1,2,3,4,5]
name_labels = ["PSP-Net","Res-U-Net","DeeplabV3+","HRNet","MEC-Net"]
flops = [2.95,10.69,14.06,23.35,15.53]

bar_width = 0.2

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.bar((np.array(x)-bar_width-0.05).tolist(), WHU_ious, bar_width, color="orange", label="WHU Dataset")
ax1.bar(x, Massachusetts_ious, bar_width, color="lightgreen", label="Massachusetts Dataset")
ax1.bar((np.array(x)+bar_width+0.05).tolist(), Inria_ious, bar_width, color="lightcoral", label="Inria Dataset")

for _x,_y in zip(x, Massachusetts_ious):
    if _x==2:
        ax1.text(_x, _y-0.5, "%.2f"%_y, color="green", ha='center', va='center', fontsize=7)
    else:
        ax1.text(_x, _y+1, "%.2f"%_y, color="green", ha='center', va='center', fontsize=7)
for _x,_y in zip(x, WHU_ious):
    ax1.text(_x-bar_width-0.05, _y+1, "%.2f"%_y, color="orange", ha='center', va='center', fontsize=7)
for _x,_y in zip(x, Inria_ious):
    ax1.text(_x+bar_width+0.05, _y+1, "%.2f"%_y, color="lightcoral", ha='center', va='center', fontsize=7)

ax1.set_ylim([60, 100])
ax1.set_ylabel('IoU(/%)')
ax1.set_xticks(x)
ax1.set_xticklabels(name_labels)
# ax1.legend(loc='upper left',ncol=2)

ax2 = ax1.twinx() # this is the important function
ax2.plot(x, flops, 'b', marker='^', label="FLOPs")
for _x,_y in zip(x, flops):
    if _x==3:
        ax2.text(_x, _y+1.5, "%.2f"%_y, color="b", ha='center', va='center', fontsize=7)
    else:
        ax2.text(_x, _y+1, "%.2f"%_y, color="b", ha='center', va='center', fontsize=7)
ax2.set_ylim([0, 30])
ax2.set_ylabel('FLOPs(/GMac)')

# ax2.legend(loc='upper right')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1+handles2, labels1+labels2, ncol = 4, columnspacing = 0.35, loc = 'upper center',borderpad = 0.3, handletextpad=0.1)
plt.savefig("demo.png",dpi=600)