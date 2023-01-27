import os.path as osp
import argparse
import numpy as np
from chester import logger

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

from PIL import Image
import time
import os

current_path = osp.dirname(__file__)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--test_num', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='RopeConfiguration')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')
    parser.add_argument('--observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--log_dir', default=osp.join(current_path,'data/cem/simple/'))
    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]
    print('arg num is %i'%args.test_num)

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = 1 #rgs.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    
    env_kwargs['observation_mode'] = args.observation_mode

    logger.configure(dir=args.log_dir, exp_name='ropeconfig')
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    obs = env.reset()
    frames = [env.get_image(args.img_size, args.img_size)]
    
    for i in range(1):
        action = env.action_space.sample()
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        next_obs, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])

    # save_dir = osp.join(logdir, 'giifs')
    # print(save_dir)
    # if osp.exists(save_dir):
    #     print('exist')
    # else:
    #     os.makedirs(save_dir, exist_ok=True)
    #     print('not exist')
if __name__ == '__main__':
    main()
    # for i in range(5):
    #     main()
    #     print(i)