import matplotlib.pyplot as plt
import numpy as np

# Data for the charts
chart1 = {"D-MMFL": {"data": 20.6, "knowledge": 25.09}, "P-MMFL": {"param": 120}}
chart2 = {"D-MMFL": {"data": 4.9, "knowledge": 5.8}, "P-MMFL": {"param": 9600}}
chart3 = {"D-MMFL": {"data": 73.2, "knowledge": 77.69}, "P-MMFL": {"param": 360}}

# Create the figure with 3 subplots
fig, axs = plt.subplots(1,3, figsize=(10, 5.5))

plt.rc('font',size=18)

# Plot the first chart on the first subplot
chart1_data = [chart1["D-MMFL"]["data"]]
chart1_knowledge = [chart1["D-MMFL"]["knowledge"]-chart1["D-MMFL"]["data"]]
chart1_param = [chart1["P-MMFL"]["param"]]

axs[0].bar(["D-MMFL"], chart1_data, label="Data")
axs[0].bar(["D-MMFL"], chart1_knowledge, bottom=chart1_data, label="Knowledge",color="red")
axs[0].bar(["P-MMFL"], chart1_param, color="orange", label="Parameters")
axs[0].set_ylabel("MB",fontsize=18)
axs[0].set_xlabel("mHealth",fontsize=19)
axs[0].tick_params(axis='x',labelsize=15)
axs[0].tick_params(axis='y',labelsize=16)
axs[0].set_ylim(0, np.max(chart1["D-MMFL"]["knowledge"]+chart1["P-MMFL"]["param"])*1)
# axs[0].legend(loc='upper left')



# Plot the second chart on the second subplot
chart2_data = [chart2["D-MMFL"]["data"]]
chart2_knowledge = [chart2["D-MMFL"]["knowledge"]-chart2["D-MMFL"]["data"]]
chart2_param = [chart2["P-MMFL"]["param"]]

axs[1].bar(["D-MMFL"], chart2_data, label="Data")
axs[1].bar(["D-MMFL"], chart2_knowledge, bottom=chart2_data, label="Knowledge",color="red")
axs[1].bar(["P-MMFL"], chart2_param, color="orange", label="Parameters")
axs[1].set_ylabel("MB",fontsize=18)
axs[1].set_xlabel("UR Fall",fontsize=19)
axs[1].tick_params(axis='x',labelsize=15)
axs[1].tick_params(axis='y',labelsize=17)
axs[1].set_ylim(1, np.max(chart2["D-MMFL"]["knowledge"]+chart2["P-MMFL"]["param"])*2)
axs[1].set_yscale("log")

# axs[1].legend(loc='upper left')

# Plot the third chart on the third subplot
chart3_data = [chart3["D-MMFL"]["data"]]
chart3_knowledge = [chart3["D-MMFL"]["knowledge"]-chart3["D-MMFL"]["data"]]
chart3_param = [chart3["P-MMFL"]["param"]]

axs[2].bar(["D-MMFL"], chart3_data, label="Data")
axs[2].bar(["D-MMFL"], chart3_knowledge, bottom=chart3_data, label="Knowledge",color="red")
axs[2].bar(["P-MMFL"], chart3_param, color="orange", label="Parameters")
axs[2].set_ylabel("MB",fontsize=18)
axs[2].set_xlabel("Opp",fontsize=19)
axs[2].tick_params(axis='x',labelsize=15)
axs[2].tick_params(axis='y',labelsize=17)
axs[2].set_ylim(0, np.max(chart3["D-MMFL"]["knowledge"]+chart3["P-MMFL"]["param"])*1)
# axs[2].legend(loc='upper left')

handles,labels=axs[2].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,1),ncol=3)
fig.subplots_adjust(wspace=0.5,hspace=0.5)


fig.savefig("figs/cost.pdf")
# Show the plots
plt.show()


