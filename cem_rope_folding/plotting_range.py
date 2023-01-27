# -*- coding: utf-8 -*-
import numpy as np
import time
import os.path as osp
import pandas as pd
import seaborn as sns

current_path = osp.dirname(__file__)

m = 10
n = 30
#############################################################################
cost_his_original = np.zeros([m,n])
ori_num_iter = np.zeros([m,n])

for i in range(m):
    j = i + 1

    temp_name = osp.join(current_path, '../data/cem/original/20220827001019/data/cost_his_' + str(j) +'.npy')
    cost_original_temp = np.load(temp_name, allow_pickle=True)
    cost_his_original[i,:] = cost_original_temp
    ori_num_iter[i,:] = np.linspace(1,n,n)
    # print('the %d-th ori data minimum is %f.' %(i, min(cost_original_temp)))

ori_cost_his_flatten = cost_his_original.flatten()
ori_num_iter = ori_num_iter.flatten()
ori_methods = np.array(['CEM(Original)']*n*m).flatten()

###############################################################################
m = 10
n = 30

cost_his_simp = np.zeros([m,n])
simp_num_iter = np.zeros([m,n])

for i in range(m):
    j = i +1

    temp_name = osp.join(current_path, '../data/cem/simple/20230124101332/data/cost_his_' + str(j) +'.npy')
    cost_simp_temp = np.load(temp_name, allow_pickle=True)
    cost_his_simp[i,:] = cost_simp_temp
    simp_num_iter[i,:] = np.linspace(1,n,n)

simp_cost_his_flatten = cost_his_simp.flatten()
simp_num_iter = simp_num_iter.flatten()
simp_methods = np.array(['CEM(Simple)']*n*m).flatten()

###############################################################################
m = 10
n = 30

cost_his_rand = np.zeros([m,n])
rand_num_iter = np.zeros([m,n])

for i in range(m):
    j = i +1

    temp_name = osp.join(current_path, '../data/cem/random/20230124123621/data/cost_his_' + str(j) +'.npy')
    cost_rand_temp = np.load(temp_name, allow_pickle=True)
    cost_his_rand[i,:] = cost_rand_temp
    rand_num_iter[i,:] = np.linspace(1,n,n)

rand_cost_his_flatten = cost_his_rand.flatten()
rand_num_iter = rand_num_iter.flatten()
rand_methods = np.array(['CEM(Random)']*n*m).flatten()

###############################################################################
## hand_crafted
m = 10
n = 30
reward_his_hand_crafted = np.zeros(m)
sudo_num_iter = np.zeros([m,n])

for i in range(m):
    j = i +1
    temp_name = osp.join(current_path, '../data/cem/hand_crafted/20220902153435/data/reward_his_' + str(j) +'.npy')

    reward_his_temp = np.load(temp_name, allow_pickle=True)
    reward_his_hand_crafted[i] = -np.max(reward_his_temp)*100
    sudo_num_iter[i,:] = np.linspace(1,n,n)

sudo_cost_his = np.ones([m,n]) * np.mean(reward_his_hand_crafted)

sudo_cost_his_flatten = sudo_cost_his.flatten()
sudo_num_iter = sudo_num_iter.flatten()
hand_crafted_methods = np.array(['Scripted']*n*m).flatten()
####################################################################################
cost_his = np.hstack([simp_cost_his_flatten, ori_cost_his_flatten, rand_cost_his_flatten, sudo_cost_his_flatten])
# print(simp_num_iter)
# print(ori_num_iter)
# print(sudo_num_iter)
num_iter = np.hstack([simp_num_iter, ori_num_iter, rand_num_iter, sudo_num_iter])
methods = np.hstack([simp_methods, ori_methods, rand_methods, hand_crafted_methods])

df = pd.DataFrame({'Cost':cost_his, 
                    'Iteration':num_iter, 
                    'Methods': methods})

###############################################################################

from matplotlib import pyplot as plt

# fig, ax = plt.subplots()
# # plt.hold(1)

# x = np.linspace(1,n,n)
# y = np.mean(cost_his_original, axis = 0)
# y_max = np.max(cost_his_original, axis = 0)
# y_min = np.min(cost_his_original, axis = 0)

# error = np.var(cost_his_original, axis = 0)

# ax.plot(x, y, 'k', color='#1B2ACC', linewidth = 3)
# ax.fill_between(x, y_min, y_max,
#     alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF', label = 'CEM-Original')
# ##################################################
# x = np.linspace(1,n,n)
# y = np.mean(cost_his_simp, axis = 0)
# y_max = np.max(cost_his_simp, axis = 0)
# y_min = np.min(cost_his_simp, axis = 0)

# error = np.var(cost_his_simp, axis = 0)

# ax.plot(x, y, 'k', color='#CC4F1B', linewidth = 3)
# ax.fill_between(x, y_min, y_max,
#     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label = 'CEM-Simple')
# ########################################################################
# ax.legend(loc='best', prop = {'size':20, 'family':'Times New Roman'})
# # ax.legend(loc='upper right', prop = {'size':20, 'family':'Times New Roman'})

# plt.xlim([1,n])
# # plt.xticks(np.arange(-1,6,1))
# plt.xticks(np.arange(1,n,4), fontsize = 30)
# plt.yticks(fontsize = 30)

# plt.xlabel('Number of iterations', fontproperties = 'Times New Roman', fontsize=30, color = 'black', weight = 'normal')
# plt.ylabel('Minimum cost', fontproperties = 'Times New Roman', fontsize=30, color = 'black', weight = 'normal')

ts = time.gmtime()
ts = time.strftime("%Y_%m_%d_%H_%M_%S_", ts)
save_path = osp.join(current_path, './data/cem/cost_history')

# plt.savefig(save_path + '_' + ts + '.png', bbox_inches = 'tight')
# # plt.savefig(save_path + '_' + ts + '.eps', dpi = 600, format = 'eps', bbox_inches = 'tight')

# plt.show()

#######################################################################
plt.figure(figsize=(8,6))
sns.set(font_scale = 2)
sns.set_style(style='white')
fig_cost_ori = sns.lineplot(x = 'Iteration', y ='Cost', hue = 'Methods', data = df, ci = 95, lw=2, palette=['#0000FF', '#FF6103', '#71C671', '#000000']) #

fig_cost_ori.lines[3].set_linestyle("--")
plt.xlim(1, 30)
plt.ylim(0, 18)

# fig_cost_ori.set_xlabel("Iteration", fontsize = 20)
# fig_cost_ori.set_ylabel("Cost", fontsize = 20)
# # get_fig.set_title("Plot", fontsize = 20)
# plt.legend(labels=["CEM-Simple","CEM-Original"], fontsize = 20)

# fig_cost_ori.set_xlabel("X-Axis")
# fig_cost_ori.set_ylabel("Y-Axis")
# fig_cost_ori.set_title("Plot")
# plt.legend(labels=["CEM-Simple","CEM-Original"])


get_fig = fig_cost_ori.get_figure()
get_fig.savefig(save_path + '_' + ts + '.png', format = 'png', dpi = 720, bbox_inches = 'tight')

plt.show()
#########################################
# sns.lineplot(data=compare_panda_wide)
# plt.show()




