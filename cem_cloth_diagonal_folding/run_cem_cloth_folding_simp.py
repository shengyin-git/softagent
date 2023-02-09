import sys
import os
import os.path as osp
sys.path.insert(1, os.getcwd())

from cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from visualize_cem import cem_make_gif
from planet.utils import transform_info
from envs.env import Env
from chester import logger
import torch
import pickle
import copy
import multiprocessing as mp
import json
import numpy as np
from softgym.registered_env import env_arg_dict
import time
import matplotlib.pyplot as plt

from utility import *

import pyflex

from PIL import Image

current_path = osp.abspath(osp.dirname(__file__))

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
        obs = env.reset()
        policy.reset()
        initial_state = env.get_state()
        action_traj = []
        action_var = []
        infos = []
        time_cost_his = []
        
        time_start=time.time()
        
        j = 0
        while j < env.horizon:
            logger.log('episode {}, step {}'.format(i, j))
            time_start = time.time()
            action, var, cost_his= policy.get_action(obs)
            time_end = time.time()
            time_cost_his.append(time_end - time_start)
            print('time cost for one optimization: ', time_cost_his[-1])
            
            is_best_reached = False
            reward_his = []
            for k in range(vv['plan_horizon'] ):
                
                if not is_best_reached:
                    current_action = action[k]
                    current_var = var[k]
                    action_traj.append(copy.copy(current_action))
                    action_var.append(current_var)
                    obs, reward, _, info = env.step(current_action)
                    infos.append(info)
                    reward_his.append(reward)
                else:
                    infos.append(info)
                    reward_his.append(reward)
                
                if abs(reward * 100 + min(cost_his)) < 0.0001:
                    is_best_reached = True
        
            j = j + vv['plan_horizon']    

        all_infos.append(infos)
        
        ###################################################################################
        ## data saving
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
            np.save(save_dir + 'action_var_' + str(vv['test_num'])  + '.npy', action_var)
        else:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir + 'action_traj_' + str(vv['test_num'])  + '.npy', action_traj)
            np.save(save_dir + 'action_var_' + str(vv['test_num'])  + '.npy', action_var)
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='cem', type=str)
    parser.add_argument('--env_name', default='SimpMeshEnv') #SimpMeshEnv RopeConfiguration RopeFlatten ClothFold
    parser.add_argument('--log_dir', default='./data/simp/cem/')
    parser.add_argument('--test_episodes', default=1, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--test_num', default=0, type=int)

    # CEM
    parser.add_argument('--max_iters', default=20, type=int) #10
    parser.add_argument('--timestep_per_decision', default=3000, type=int) #default=21000 120000
    parser.add_argument('--use_mpc', default=False, type=bool)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=False, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)
    parser.add_argument('--env_kwargs_num_variations', default=1, type=int)
    parser.add_argument('--env_kwargs_num_picker', default=1, type=int)
    
    #
    parser.add_argument('--env_kwargs_horizon', default=15, type=int) #15
    parser.add_argument('--env_kwargs_action_repeat', default=8, type=int) # 8
    # parser.add_argument('--env_kwargs_headless', default=0, type=int)
    parser.add_argument('--env_kwargs_use_cached_states', default=True, type=bool)
    parser.add_argument('--env_kwargs_save_cached_states', default=False, type=bool)
    
    # parser.add_argument('--env_kwargs_use_simplified_key_point', default=Ture, type=bool)

    reorder_key_point_idx, nodes_idx, node_flat_positions, node_folded_positions, edges = \
            pre_data_process(flatten_pos_path = './data/simp_model/particle_pos_ini.npy',\
                        folded_pos_path='./data/simp_model/particle_pos_final.npy',\
                        simp_model_path='./data/simp_model/simped_model.npy',\
                        key_point_idx_path='./data/simp_model/key_point_index.npy',\
                        second_model_path = None)
    node_set = node_flat_positions
    edge_set = edges
    parser.add_argument('--env_kwargs_node_set', default=node_set, type=np.array)
    parser.add_argument('--env_kwargs_edge_set', default=edge_set, type=np.array)
    parser.add_argument('--env_kwargs_nodes_idx', default=nodes_idx, type=np.array)
    parser.add_argument('--env_kwargs_goal_key_pos', default=node_folded_positions, type=np.array)
    parser.add_argument('--env_kwargs_key_point_idx', default=reorder_key_point_idx, type=np.array)

    args = parser.parse_args()
    run_task(args.__dict__, args.log_dir, args.exp_name)

if __name__ == '__main__':
    main()
