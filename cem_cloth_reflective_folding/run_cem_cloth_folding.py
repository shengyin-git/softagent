import os.path as osp
upper_path = osp.abspath(osp.join(osp.abspath('.'),'..'))
import sys
sys.path.insert(1, upper_path)

from cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from visualize_cem import cem_make_gif
from planet.utils import transform_info
from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import multiprocessing as mp
import json
import numpy as np
from softgym.registered_env import env_arg_dict
import time
import matplotlib.pyplot as plt

import pyflex

from PIL import Image

current_path = osp.dirname(__file__)
    
def plotting(data_x, data_y, save_dir, x_label, y_label, line_color = 'red', line_width = 2,  line_marker = 'o', line_label = 'cost history'):
    
    fig = plt.figure(num=1, figsize = (4,4))
    plt.plot(data_x, data_y, c = line_color, linewidth = line_width, marker = line_marker, label = line_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
        
    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)

    save_dir =  osp.join(current_path, 'images/')
    if os.path.exists(save_dir):
        plt.savefig(save_dir + ts + '.png')
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + ts + '.png')
        
    plt.show()

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args

def get_reward(initial_pos, current_pos):
        num_particles = 100*100
        particle_grid_idx = np.array(list(range(num_particles))).reshape(100, 100) 

        x_split = 100 // 2

        fold_group_a = particle_grid_idx[:, :x_split].flatten()
        fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        pos_group_a = current_pos[fold_group_a]
        pos_group_b = current_pos[fold_group_b]

        pos_group_b_init = initial_pos[fold_group_b]

        curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + \
                    1.2 * np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))

        reward = -curr_dist
        return reward

def run_task(vv, log_dir, exp_name):
    print(vv['test_num'])
    mp.set_start_method('spawn')
    env_name = vv['env_name']
    vv['algorithm'] = 'CEM'
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv['plan_horizon'] = vv['env_kwargs_horizon'] #cem_plan_horizon[env_name]  # Planning horizon

    vv['population_size'] = vv['timestep_per_decision'] // vv['max_iters']
    if vv['use_mpc']:
        vv['population_size'] = vv['population_size'] // vv['plan_horizon']
    vv['num_elites'] = vv['population_size'] // 10 #10
    vv = update_env_kwargs(vv)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Dump parameters
    # with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
    #     json.dump(vv, f, indent=2, sort_keys=True)

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'

    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'max_episode_length': 200,
                  'action_repeat': 1,  # Action repeat for env wrapper is 1 as it is already inside the env
                  'bit_depth': 8,
                  'image_dim': None,
                  'env_kwargs': vv['env_kwargs']}
    
    env = env_class(**env_kwargs)

    policy = CEMPolicy(env, env_class, env_kwargs, vv['use_mpc'], plan_horizon=vv['plan_horizon'], max_iters=vv['max_iters'],
                       population_size=vv['population_size'], num_elites=vv['num_elites'])
    # Run policy
    initial_states, action_trajs, configs, all_infos = [], [], [], []
    for i in range(vv['test_episodes']):
        logger.log('episode ' + str(i))
        #################################################
        # obs = env.reset()
        # initial_pos_ = pyflex.get_positions().reshape((-1, 4))[:, :3]
        # env.action_tool.reset_previous_action_space()

        # saved_action_path = './data/action_traj_1.npy'
        # action_seq = np.load(saved_action_path)
        # num_actions = len(action_seq[:,0])

        # for k in range(num_actions):
        #     current_action = action_seq[k]
        #     obs, reward, _, info = env.step(current_action)
        #     current_pos_ = pyflex.get_positions().reshape((-1, 4))[:, :3]
        #     print(get_reward(initial_pos_, current_pos_))

        obs = env.reset()
        #################################################
        policy.reset()
        initial_state = env.get_state()
        action_traj = []
        infos = []
        time_cost_his = []
        
        time_start=time.time()
        
        j = 0
        while j < env.horizon:
            logger.log('episode {}, step {}'.format(i, j))
            time_start = time.time()
            action, cost_his= policy.get_action(initial_state) #action, cost_his= policy.get_action(obs)
            time_end = time.time()
            time_cost_his.append(time_end - time_start)
            print('time cost for one optimization: ', time_cost_his[-1])
            
            is_best_reached = False
            reward_his = []
            for k in range(vv['plan_horizon'] ):
                
                if not is_best_reached:
                    current_action = action[k]

                    action_traj.append(copy.copy(current_action))
                    obs, reward, _, info = env.step(current_action)
                    infos.append(info)
                    reward_his.append(reward)
                else:
                    infos.append(info)
                    reward_his.append(reward)
                
                if abs(reward * 100 + min(cost_his)) < 0.0001:
                    is_best_reached = True
        
            j = j + vv['plan_horizon']    
            
        ## plotting
        # plotting(data_x = np.linspace(1,len(cost_his),len(cost_his)), data_y = cost_his, save_path = './data/cem/cloth_folding/figs/cost_history', \
        #          x_label = 'number of iterations', y_label = 'minimum cost per iteration', line_color = 'red', line_width = 2,  line_marker = 'o', \
        #              line_label = 'cost history')
            
        # plotting(data_x = np.linspace(1,len(reward_his),len(reward_his)), data_y = reward_his, save_path = './data/cem/cloth_folding/figs/reward_history', \
        #          x_label = 'number of motion steps', y_label = 'reward per step', line_color = 'red', line_width = 2,  line_marker = 'o', \
        #              line_label = 'reward curve')

        all_infos.append(infos)
        
        ts = time.gmtime()
        ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
        
        save_dir = osp.join(logdir, 'data/')
        if osp.exists(save_dir):     
            np.save(save_dir + 'all_infos_' + str(vv['test_num']) + '.npy', all_infos)
            np.save(save_dir + 'cost_his_' + str(vv['test_num'])  + '.npy', cost_his)
            np.save(save_dir + 'time_cost_his_' + str(vv['test_num'])  + '.npy', time_cost_his)
            np.save(save_dir + 'reward_his_' + str(vv['test_num'])  + '.npy', reward_his)
        else:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir + 'all_infos_' + str(vv['test_num'])  + '.npy', all_infos)
            np.save(save_dir + 'cost_his_' + str(vv['test_num'])  + '.npy', cost_his)
            np.save(save_dir + 'time_cost_his_' + str(vv['test_num'])  + '.npy', time_cost_his)
            np.save(save_dir + 'reward_his_' + str(vv['test_num'])  + '.npy', reward_his)

        save_dir = osp.join(logdir, 'traj/')
        if osp.exists(save_dir):  
            np.save(save_dir + 'action_traj_' + str(vv['test_num'])  + '.npy', action_traj)
        else:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir + 'action_traj_' + str(vv['test_num'])  + '.npy', action_traj)
        print(min(cost_his))
        print(max(reward_his))
        
        initial_states.append(initial_state.copy())
        action_trajs.append(action_traj.copy())
        configs.append(env.get_current_config().copy())

        # Log for each episode
        transformed_info = transform_info([infos])
        for info_name in transformed_info:
            logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
            logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
            logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
        logger.dump_tabular()

    # Dump trajectories
    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)

def get_rand_int(low, high, size):
    rand_int = []
    i = 0
    while i < size:
        rand_temp = np.random.randint(low=low, high=high)
        if rand_temp not in rand_int:
            rand_int.append(rand_temp)
            i = i + 1
    return np.array(rand_int)

def get_rand_edge_idx(width, height, size):
    edge_idx = []
    
    for i in range(height):
        if i == 0:
            for j in range(width):
                edge_idx.append(j)
        elif i == height - 1:
            temp = width*(height-1)
            for j in range(width):
                edge_idx.append(temp+j)
        else:
            edge_idx.append(i*width)
            edge_idx.append((i+1)*width-1)
    
    size_edge_idx = len(edge_idx)

    rand_int = []
    i = 0
    while i < size:
        rand_temp = edge_idx[np.random.randint(low=0, high=size_edge_idx)]
        if rand_temp not in rand_int:
            rand_int.append(rand_temp)
            i = i + 1
    return np.array(rand_int)

cem_plan_horizon = {
    'PassWater': 7,
    'PourWater': 40,
    'PourWaterAmount': 40,
    'ClothFold': 15, # 15
    'ClothFoldCrumpled': 30,
    'ClothFoldDrop': 30,
    'ClothFlatten': 15,
    'ClothDrop': 15,
    'RopeFlatten': 15,
    'RopeConfiguration': 20, #15,
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='cem', type=str)
    parser.add_argument('--env_name', default='ClothReflectiveFold') #RopeConfiguration RopeFlatten ClothFold
    parser.add_argument('--log_dir', default='./data/cem/original/')
    parser.add_argument('--test_episodes', default=1, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--test_num', default=0, type=int)

    # CEM
    parser.add_argument('--max_iters', default=30, type=int) #10
    parser.add_argument('--timestep_per_decision', default=10000, type=int) #default=21000 120000
    parser.add_argument('--use_mpc', default=False, type=bool)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=False, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)
    parser.add_argument('--env_kwargs_num_variations', default=1, type=int)
    parser.add_argument('--env_kwargs_num_picker', default=1, type=int)
    
    # 4 24
    # 8 12
    # 15 8
    # 4 28
    # 4 32
    parser.add_argument('--env_kwargs_horizon', default=4, type=int) #15
    parser.add_argument('--env_kwargs_action_repeat', default=32, type=int) # 8
    # parser.add_argument('--env_kwargs_headless', default=0, type=int)
    parser.add_argument('--env_kwargs_use_cached_states', default=True, type=bool)
    parser.add_argument('--env_kwargs_save_cached_states', default=False, type=bool)
    
    key_point_idx = np.array([0, 24, 49, 99, 9902, 9924, 9947, 9997])
    #get_rand_edge_idx(width= 100, height = 100, size = 8) #
    # print(key_point_idx)
    # get_rand_int(low=0, high=8100, size=4)
    #np.random.randint(low=0, high=40, size=2)##
    parser.add_argument('--env_kwargs_key_point_idx', default=key_point_idx, type=np.array)

    args = parser.parse_args()
    run_task(args.__dict__, args.log_dir, args.exp_name)

if __name__ == '__main__':
    main()
