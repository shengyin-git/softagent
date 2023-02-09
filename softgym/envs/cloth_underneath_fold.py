import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object

class ClothUnderneathFoldEnv(ClothEnv):
    def __init__(self, key_point_idx, cached_states_path='cloth_underneath_fold_init_states.pkl', **kwargs):
        self.key_point_idx = key_point_idx
        self.fold_group_a = self.fold_group_b = self.fold_group_c = self.fold_group_d = None
        self.init_pos, self.prev_dist = None, None
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']: # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()
            angle = (np.random.random() - 0.5) * np.pi / 2
            self.rotate_particles(angle)

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1, p2), :3] # Was changed from from p1, p4.
            middle_point = np.mean(key_point_pos, axis=0)
            self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])
            self.action_tool.reset_action_space(self.key_point_idx)

            # picker_low = self.action_tool.picker_low
            # picker_high = self.action_tool.picker_high
            # offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
            # picker_low[0] += offset_x
            # picker_high[0] += offset_x
            # picker_high[0] += 1.0
            # self.action_tool.update_picker_boundary(picker_low, picker_high)

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split_0 = cloth_dimx // 2
        x_split_1 = x_split_0 // 2

        self.fold_group_a = np.flip(particle_grid_idx[:, :x_split_1], axis = 1).flatten()        
        self.fold_group_b = particle_grid_idx[:, x_split_1:x_split_0].flatten()
        self.fold_group_c = np.flip(np.flip(particle_grid_idx, axis=1)[:, x_split_1:x_split_0], axis=1).flatten()
        self.fold_group_d = np.flip(np.flip(particle_grid_idx, axis=1)[:, :x_split_0], axis = 1).flatten()
        self.fold_group_e = np.flip(np.flip(particle_grid_idx, axis=1)[:, :x_split_1], axis = 1).flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self._set_to_folded()
        pyflex.step()

        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']

        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            print(self.action_tool.next_action)
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_c = pos[self.fold_group_c]
        pos_group_d = pos[self.fold_group_d]
        pos_group_e = pos[self.fold_group_e]

        pos_group_b_init = self.init_pos[self.fold_group_b]
        pos_group_d_init = self.init_pos[self.fold_group_d]
        pos_group_c_init = self.init_pos[self.fold_group_c]
        pos_group_e_init = self.init_pos[self.fold_group_e]
        norminal_a = pos_group_c_init + pos_group_c_init - pos_group_e_init
        curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + \
                    1.2 * np.mean(np.linalg.norm(pos_group_a - norminal_a, axis=1)) + \
                    1.2 * np.mean(np.linalg.norm(pos_group_d - pos_group_d_init, axis=1))
        reward = -curr_dist
        return reward

    # def compute_reward(self, action=None, obs=None, set_prev_reward=False):
    #     """
    #     The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
    #     particle in group a and the crresponding particle in group b
    #     :param pos: nx4 matrix (x, y, z, inv_mass)
    #     """
    #     config = self.get_current_config()
    #     num_particles = np.prod(config['ClothSize'], dtype=int)
    #     particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

    #     cloth_dimx = config['ClothSize'][0]
    #     x_split_0 = cloth_dimx // 2
    #     x_split_1 = x_split_0 // 2

    #     self.fold_group_a = particle_grid_idx[:, :x_split_1].flatten()
        
    #     self.fold_group_b = np.flip(particle_grid_idx[:, x_split_1:x_split_0], axis=1).flatten()

    #     self.fold_group_c = np.flip(np.flip(particle_grid_idx, axis=1)[:, x_split_1:x_split_0], axis=1).flatten()
    #     self.fold_group_d = np.flip(particle_grid_idx, axis=1)[:, :x_split_0].flatten()
        
    #     pos = pyflex.get_positions()
    #     pos = pos.reshape((-1, 4))[:, :3]
    #     pos_group_a = pos[self.fold_group_a]
    #     pos_group_b = pos[self.fold_group_b]
    #     pos_group_c = pos[self.fold_group_c]
    #     pos_group_d = pos[self.fold_group_d]

    #     pos_group_b_init = self.init_pos[self.fold_group_b]
    #     pos_group_d_init = self.init_pos[self.fold_group_d]

    #     # pos_target_group_d = self.flattened_particle_pos[self.fold_group_d]
    #     # pos_target_group_c = self.flattened_particle_pos[self.fold_group_c]

    #     curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + \
    #                 np.mean(np.linalg.norm(pos_group_b - pos_group_c, axis=1)) + \
    #                 1.2 * np.mean(np.linalg.norm(pos_group_d - pos_group_d_init, axis=1))
    #     reward = -curr_dist
    #     return reward

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        performance = -group_dist - 1.2 * fixation_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        info = {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init),
            'neg_group_dist': -group_dist,
            'neg_fixation_dist': -fixation_dist
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def _set_to_folded(self):
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]

        ## 1/2
        x_split_0 = cloth_dimx // 2
        x_split_1 = x_split_0 // 2

        fold_group_a = particle_grid_idx[:, :x_split_1].flatten()
        
        fold_group_b = np.flip(particle_grid_idx[:, x_split_1:x_split_0], axis=1).flatten()

        fold_group_c = np.flip(np.flip(particle_grid_idx, axis=1)[:, x_split_1:x_split_0], axis=1).flatten()
        fold_group_d = np.flip(particle_grid_idx, axis=1)[:, :x_split_0].flatten()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[fold_group_b, :] = curr_pos[fold_group_c, :].copy()
        curr_pos[fold_group_b, 1] += 0.05   
        
        curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
        curr_pos[fold_group_a, 1] += 0.05

        # ## 1/3
        # x_split_0 = cloth_dimx // 2
        # x_split_1 = x_split_0 // 3

        # fold_group_a = particle_grid_idx[:, :x_split_1].flatten()
        
        # fold_group_b = np.flip(particle_grid_idx[:, x_split_1:x_split_1*2], axis=1).flatten()
        # fold_group_c = np.flip(particle_grid_idx[:, x_split_1:x_split_0], axis=1).flatten()

        # fold_group_d = np.flip(np.flip(particle_grid_idx, axis=1)[:, x_split_1:x_split_0], axis=1).flatten()
        # fold_group_e = np.flip(particle_grid_idx, axis=1)[:, :x_split_0].flatten()

        # curr_pos = pyflex.get_positions().reshape((-1, 4))
        # curr_pos[fold_group_c, :] = curr_pos[fold_group_d, :].copy()
        # curr_pos[fold_group_c, 1] += 0.05   
        
        # curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
        # curr_pos[fold_group_a, 1] += 0.05

        # ## original side folded state
        # x_split = cloth_dimx // 2
        # fold_group_a = particle_grid_idx[:, :x_split].flatten()        
        # fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()
        # curr_pos = pyflex.get_positions().reshape((-1, 4))
        # curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
        # curr_pos[fold_group_a, 1] += 0.05

        pyflex.set_positions(curr_pos)
        for i in range(15):
            pyflex.step()
        # return self._get_info()['performance']

    def _set_to_final_folded(self):
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]

        ## 1/2
        x_split_0 = cloth_dimx // 2
        x_split_1 = x_split_0 // 2

        fold_group_a = np.flip(particle_grid_idx[:, :x_split_1], axis = 1).flatten()
        
        fold_group_b = particle_grid_idx[:, x_split_1:x_split_0].flatten()

        fold_group_c = np.flip(np.flip(particle_grid_idx, axis=1)[:, x_split_1:x_split_0], axis=1).flatten()
        fold_group_d = np.flip(np.flip(particle_grid_idx, axis=1)[:, :x_split_0], axis = 1).flatten()
        fold_group_e = np.flip(np.flip(particle_grid_idx, axis=1)[:, :x_split_1], axis = 1).flatten()
     
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        trnas_matrix = curr_pos[fold_group_c,:] - curr_pos[fold_group_e,:]
        curr_pos[fold_group_a, :] = trnas_matrix + curr_pos[fold_group_c,:]
        curr_pos[fold_group_b, :] = curr_pos[fold_group_a, :].copy()
        curr_pos[fold_group_b, 1] += 0.025 

        pyflex.set_positions(curr_pos)
        for i in range(30):
            pyflex.step()
        # return self._get_info()['performance']

    def get_goal_image(self, camera_height, camera_width):
        # config = self.get_current_config()
        config = self.get_side_view_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]

        ## 1/2
        x_split_0 = cloth_dimx // 2
        x_split_1 = x_split_0 // 2

        fold_group_a = np.flip(particle_grid_idx[:, :x_split_1], axis = 1).flatten()
        
        fold_group_b = particle_grid_idx[:, x_split_1:x_split_0].flatten()

        fold_group_c = np.flip(np.flip(particle_grid_idx, axis=1)[:, x_split_1:x_split_0], axis=1).flatten()
        fold_group_d = np.flip(np.flip(particle_grid_idx, axis=1)[:, :x_split_0], axis = 1).flatten()
        fold_group_e = np.flip(np.flip(particle_grid_idx, axis=1)[:, :x_split_1], axis = 1).flatten()
     
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        trnas_matrix = curr_pos[fold_group_c,:] - curr_pos[fold_group_e,:]
        curr_pos[fold_group_a, :] = trnas_matrix + curr_pos[fold_group_c,:]
        curr_pos[fold_group_b, :] = curr_pos[fold_group_a, :].copy()
        curr_pos[fold_group_b, 1] += 0.025 

        pyflex.set_positions(curr_pos)
        for i in range(30):
            pyflex.step()
            
        default_config = self.get_default_config()
        self.update_camera('default_camera', default_config['camera_params']['default_camera']) 
        self.action_tool.reset([0., -1., 0.]) # hide picker
        goal_img = self.get_image(camera_height, camera_width)
        return goal_img 

    def get_initial_image(self, camera_height, camera_width):
        all_positions = pyflex.get_positions().reshape([-1, 4])
        initial_pos =  self.cached_init_states[0]['particle_pos'].reshape([-1,4])[:,:3]
        all_positions[:,0:3] = initial_pos.copy()
        pyflex.set_positions(all_positions)
        default_config = self.get_default_config()
        self.update_camera('default_camera', default_config['camera_params']['default_camera']) 
        self.action_tool.reset([0., -1., 0.]) # hide picker
        goal_img = self.get_image(camera_height, camera_width)
        return goal_img 


