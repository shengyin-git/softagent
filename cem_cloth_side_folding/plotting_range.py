# -*- coding: utf-8 -*-
import numpy as np
import time
import os.path as osp
import pandas as pd
import seaborn as sns

current_path = osp.dirname(__file__)
###########################################################################
m = 10
n = 30

cost_his_original = np.zeros([m,n])
ori_num_iter = np.zeros([m,n])

for i in range(m):
    j = i + 1
    temp_name = osp.join(current_path, '../data/cem/original/20220906130809/data/cost_his_' + str(j) +'.npy')
    cost_original_temp = np.load(temp_name, allow_pickle=True)
    
    cost_his_original[i,:] = cost_original_temp
    ori_num_iter[i,:] = np.linspace(1,n,n)

ori_cost_his_flatten = cost_his_original.flatten()
ori_num_iter = ori_num_iter.flatten()
ori_methods = np.array(['Original']*n*m).flatten()

###############################################################################
m = 10
n = 30

cost_his_simp = np.zeros([m,n])
simp_num_iter = np.zeros([m,n])

for i in range(m):
    j = i +1
    temp_name = osp.join(current_path, '../data/cem/simple/20220904220252/data/cost_his_' + str(j) +'.npy')
    cost_simp_temp = np.load(temp_name, allow_pickle=True)
    cost_his_simp[i,:] = cost_simp_temp
    simp_num_iter[i,:] = np.linspace(1,n,n)

simp_cost_his_flatten = cost_his_simp.flatten()
simp_num_iter = simp_num_iter.flatten()
simp_methods = np.array(['Simplified']*n*m).flatten()

################################################################################
m = 10
n = 30

cost_his_rand = np.zeros([m,n])
rand_num_iter = np.zeros([m,n])

for i in range(m):
    j = i +1
    temp_name = osp.join(current_path, '../data/random/cem/20230206104142/data/cost_his_' + str(j) +'.npy')
    cost_rand_temp = np.load(temp_name, allow_pickle=True)
    cost_his_rand[i,:] = cost_rand_temp
    rand_num_iter[i,:] = np.linspace(1,n,n)

rand_cost_his_flatten = cost_his_rand.flatten()
rand_num_iter = rand_num_iter.flatten()
rand_methods = np.array(['Random']*n*m).flatten()

###############################################################################
## hand_crafted
m = 10
n = 30
reward_his_hand_crafted = np.zeros(m)
sudo_num_iter = np.zeros([m,n])

for i in range(m):
    j = i +1
    temp_name = osp.join(current_path, '../data/cem/hand_crafted/20220902163202/data/reward_his_' + str(j) +'.npy')

    reward_his_temp = np.load(temp_name, allow_pickle=True)
    reward_his_hand_crafted[i] = -np.max(reward_his_temp)*100
    sudo_num_iter[i,:] = np.linspace(1,n,n)

sudo_cost_his = np.ones([m,n]) * np.mean(reward_his_hand_crafted)

sudo_cost_his_flatten = sudo_cost_his.flatten()
sudo_num_iter = sudo_num_iter.flatten()
hand_crafted_methods = np.array(['Scripted']*n*m).flatten()

######################################################################################
cost_his = np.hstack([simp_cost_his_flatten, ori_cost_his_flatten, rand_cost_his_flatten, sudo_cost_his_flatten])
num_iter = np.hstack([simp_num_iter, ori_num_iter, rand_num_iter, sudo_num_iter])
methods = np.hstack([simp_methods, ori_methods, rand_methods, hand_crafted_methods])

df = pd.DataFrame({'Cost':cost_his, 
                    'Iteration':num_iter, 
                    'Methods': methods})

###############################################################################
from matplotlib import pyplot as plt

ts = time.gmtime()
ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
save_path = osp.join(current_path, 'data/cem/cost_history')
#######################################################################
plt.figure(figsize=(8,6))
sns.set(font_scale = 2)
sns.set_style(style='white')

fig_cost_ori = sns.lineplot(x = 'Iteration', y ='Cost', hue = 'Methods', data = df, ci = 95, lw=2, palette=['#0000FF', '#FF6103', '#71C671', '#000000']) #

fig_cost_ori.lines[3].set_linestyle("--")
plt.xlim(1, 30)
plt.ylim(1, 30)

get_fig = fig_cost_ori.get_figure()
get_fig.savefig(save_path + '_' + ts + '.png', dpi = 720, format = 'png', bbox_inches = 'tight')

plt.show()




