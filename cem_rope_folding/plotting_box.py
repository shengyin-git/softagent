import numpy as np
import time
import os.path as osp

current_path = osp.dirname(__file__)

## original action space
m = 10
n = 30

cost_his_original = np.zeros([m,n])

for i in range(m):
    j = i + 1
    #cost_his_original_long_
    #cost_his_original_
    temp_name = osp.join(current_path, 'data/cem/original/data/cost_his_' + str(j) +'.npy')
    cost_original_temp = np.load(temp_name, allow_pickle=True)
    cost_his_original[i,:] = cost_original_temp

## simplified action space

m = 10
n = 30

cost_his_simp = np.zeros([m,n])

for i in range(m):
    j = i +1
    #cost_his_simple_long_
    #cost_his_simple_
    temp_name = osp.join(current_path, 'data/cem/simple/data/cost_his_' + str(j) +'.npy')
    cost_simp_temp = np.load(temp_name, allow_pickle=True)
    cost_his_simp[i,:] = cost_simp_temp
## data
data_1 = cost_his_original[:,-1]
data_2 = cost_his_simp[:,-1]
data = [data_1, data_2]

## plotting
from matplotlib import pyplot as plt

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)


bp = ax.boxplot(data, patch_artist = True,
                notch ='True', vert = 0)

colors = ['#0000FF', '#00FF00']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")

for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)

for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)

for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)

ax.set_yticklabels(['data_1', 'data_2'])

plt.title("Customized box plot")

# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()

plt.show()

