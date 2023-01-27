import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import Picker
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy
from softgym.utils.pyflex_utils import center_object


class TestNewEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, horizon=75, render_mode='particle', picker_radius=0.02, **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['point_cloud', 'cam_rgb', 'key_point']
        assert action_mode in ['picker', 'sawyer', 'franka']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.num_picker = num_picker

        if action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, picker_threshold=0.005, 
            particle_radius=0.025/2, picker_low=(-0.8, 0., -0.8), picker_high=(0.8, 0.3, 0.8))
            self.action_space = self.action_tool.action_space
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)

        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = 10 * 3
            else:
                max_particles = 41
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim
            if action_mode in ['picker']:
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        self.horizon = horizon
        # print("init of rope new env done!")
        cached_states_path='test_env_init_states.pkl'
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        if config is None:
            config = self.get_default_config()
        default_config = config    
        for i in range(num_variations):
            config = deepcopy(default_config)
            self.set_scene(config)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            # random_pick_and_place(pick_num=4, pick_scale=0.005)
            center_object()
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        # node_set = np.array([[0, 0, 0],[0.5,0,0]])
        # edge_set = np.array([[0,1]])

        # node_set = np.array([[0, 0, 0.2],[-0.2, 0, -0.1],[0.2, 0, -0.1]])
        # edge_set = np.array([[0,1],[0,2],[1,2]])
        # edge_set = np.array([[0,1],[1,2],[2,0]])

        # node_set = np.array([[-0.2, 0, 0.2],[0.2, 0, 0.2],[-0.2, 0, -0.2],[0.2, 0, -0.2]])
        # edge_set = np.array([[0,1],[0,2],[0,3],[1,3],[2,3]])

        # node_set = np.array([[-0.16196716, 0.005, -0.4064387], [0.11936501, 0.005, -0.28543764],
        #                         [0.4064387, 0.005, -0.16196716], [-0.4064387, 0.005, 0.16196716],
        #                         [-0.12510654, 0.005, 0.28296822], [0.16196716, 0.005, 0.4064387]])
        # edge_set = np.array([[0,3],[3,4],[4,0],[0,1],[1,4],[4,5],[5,1],[1,2],[2,5]])

        node_set = np.array([[-0.16196716, 0.005, -0.4064387],[0.4064387, 0.005, -0.16196716],
                                [-0.4064387, 0.005, 0.16196716], [0.16196716, 0.005, 0.4064387]])
        edge_set = np.array([[2,0],[0,1],[1,2],[2,3],[3,1]])
        
        config = {
            'node_set': node_set,
            'edge_set': edge_set,
            'stretchstiffness': 0.9,
            'bendstiffness': 0.8,
            'shearstiffness': 0.8,
            'rest_length': 0.025/2,
            'mass': 50, # 20
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0, 1.95, 0]),
                                   'angle': np.array([0 * np.pi, -90 / 180. * np.pi, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            # keypoint_pos = particle_pos[self.key_point_indices, :3] #particle_pos[:, :3] # 
            pos = particle_pos#keypoint_pos.flatten()
            # more_info = np.array([self.rope_length, self._get_endpoint_distance()])
            # pos = np.hstack([more_info, pos])
            # print("in _get_obs, pos is: ", pos)

        # if self.action_mode in ['sphere', 'picker']:
        #     shapes = pyflex.get_shape_states()
        #     shapes = np.reshape(shapes, [-1, 14])
        #     pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def _get_key_point_idx(self, num=None):
        indices = [0]
        interval = (num - 2) // 8
        for i in range(1, 9):
            indices.append(i * interval)
        indices.append(num - 1)

        return indices

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        else:
            render_mode = 2

        camera_params = config['camera_params'][config['camera_name']]
        params = np.array(
            [config['rest_length'], config['stretchstiffness'], config['bendstiffness'], config['shearstiffness'], config['mass'],
                *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'],
                len(config['node_set']), len(config['edge_set']),
                *config['node_set'].flatten(), *config['edge_set'].flatten(), render_mode]
            )

        env_idx = 5

        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (params, robot_params)
            pyflex.set_scene(env_idx, params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, params, 0)

        num_particles = pyflex.get_n_particles()
        # print("with {} segments, the number of particles are {}".format(config['segment'], num_particles))
        # exit()
        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def _get_info(self):
        return {}

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def _reset(self):

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])
        print('reset finished!')

        return 0 #self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""
        # curr_endpoint_dist = self._get_endpoint_distance()
        # curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        # r = curr_distance_diff
        return 0



if __name__ == '__main__':
    env = TestNewEnv(observation_mode='key_point',
                  action_mode='picker',
                  num_picker=2,
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  num_variations=10,
                  use_cached_states=False,
                  save_cached_states=False,
                  deterministic=False)
    env.reset(config=env.get_default_config())
    for i in range(1000):
        print(i)
        print("right before pyflex step")
        pyflex.step()
        print("right after pyflex step")
        print("right before pyflex render")
        pyflex.render()
        print("right after pyflex render")
