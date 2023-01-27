from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import time
import os

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

def pre_data_process(flatten_pos_path = './data/simp_model/particle_pos_ini.npy',\
                        folded_pos_path='./data/simp_model/particle_pos_final.npy',\
                        simp_model_path='./data/simp_model/simped_model.npy',\
                        second_model_path = None):
    flat_pos = np.load(osp.join(current_path, flatten_pos_path), allow_pickle=True)
    folded_pos = np.load(osp.join(current_path, folded_pos_path), allow_pickle=True)
    simp_model = np.load(osp.join(current_path, simp_model_path), allow_pickle=True).item()

    key_point_idx = simp_model['simp_key_point_idx']
    triangles = simp_model['simp_triangles']

    edges = []
    num_triangles = len(triangles)

    for i in range(num_triangles):
        triangle = triangles[i]
        triangle = np.hstack((triangle, triangle[0]))
        for j in range(3):
            edge = [triangle[j], triangle[j+1]]
            if edge not in edges and edge[::-1] not in edges:
                edges.append(edge)

    num_edges = len(edges)
    nodes_idx = []
    for i in range(num_edges):
        node_start = edges[i][0]
        node_end = edges[i][1]
        if node_start not in nodes_idx:
            nodes_idx.append(int(node_start))
        if node_start not in nodes_idx:
            nodes_idx.append(int(node_end))

    reorder_key_point_idx = np.array(key_point_idx)[nodes_idx]
    node_flat_positions = flat_pos[key_point_idx] ## this one is not reordered
    node_folded_positions = folded_pos[key_point_idx]

    if second_model_path is not None:
        _reorder_key_point_idx, _node_flat_positions, _node_folded_positions, _edges = \
            pre_data_process(folded_pos_path = './data/simp_model/particle_pos_final.npy',\
                            simp_model_path = second_model_path,\
                            second_model_path = None)

        ## update the reorder_key_point_idx
        num_key_points = len(reorder_key_point_idx)
        for i in range(num_key_points):
            distance = np.sqrt(np.sum(np.asarray(node_flat_positions[i] - _node_flat_positions)**2, axis = 1))
            print(distance)
            index_distance = np.transpose(np.vstack((_reorder_key_point_idx, distance))) 
            index_distance = index_distance[np.argsort(index_distance[:,1])]
            reorder_key_point_idx[i] = index_distance[0,0]

        node_flat_positions = _node_flat_positions
        node_folded_positions = _node_folded_positions
        edges = _edges

    return reorder_key_point_idx, node_flat_positions, node_folded_positions, edges
    
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