import torch
from typing import Any, Dict, Optional, List


def convert_state_to_tensor(state: Dict):
    result = []
    if 'position' in state:
        result += [state['position']['x'], state['position']['y'], state['position']['z']]
    if 'rotation' in state:
        result += [state['rotation']['x'], state['rotation']['y'], state['rotation']['z']]
    return torch.Tensor(result)


def diff_position(state_goal, state_curr):
    p1 = state_goal['position']
    p2 = state_curr['position']
    result = {k:abs(p1[k] - p2[k]) for k in p1.keys()}
    return result

#LATER_TODO this is not optimized, there are very few rotation matrices for the agent that we can consider, we can just cache them
import numpy as np
# from utils.quaternion_util import quaternion_to_rotation_matrix
from scipy.spatial.transform import Rotation as R

def make_rotation_matrix(position, rotation):
    result = np.zeros((4,4))
    r = R.from_euler('xyz', [rotation['x'], rotation['y'], rotation['z']], degrees=True)
    result[:3, :3] = r.as_matrix()
    result[3, 3] = 1
    result[:3, 3] = [position['x'], position['y'], position['z']]
    return result

def inverse_rot_trans_mat(mat):
    # not sure if this actually works
    # mat = mat + 0.
    # mat[:3, :3] = np.linalg.inv(mat[:3, :3])
    # mat[:3, 3] *= -1
    # assert mat.shape == np.shape((3,3))
    mat = np.linalg.inv(mat)
    return mat

def position_rotation_from_mat(matrix):
    result = {'position':None, 'rotation':None}
    rotation = R.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)
    rotation_dict = {'x': rotation[0], 'y': rotation[1], 'z': rotation[2]}
    result['rotation'] = rotation_dict
    position = matrix[:3, 3]
    result['position'] = {'x': position[0], 'y': position[1], 'z': position[2]}
    return result

def convert_world_to_agent_coordinate(world_obj, agent_state):
    agent_rotation_matrix = make_rotation_matrix(agent_state['position'], agent_state['rotation'])
    agent_translation = agent_rotation_matrix[:3, 3]
    inverse_agent_rotation = inverse_rot_trans_mat(agent_rotation_matrix[:3, :3])
    obj_matrix = make_rotation_matrix(world_obj['position'], world_obj['rotation'])
    obj_translation = np.matmul(inverse_agent_rotation, (obj_matrix[:3, 3] - agent_translation))
    #KIANA add rotation later
    obj_matrix[:3, 3] = obj_translation
    result = position_rotation_from_mat(obj_matrix)
    return result

def test_translation_stuff():
    agent_coordinate = {
        'position':{
            'x': 1, 'y':0, 'z':2
        },
        'rotation':{
            'x': 0, 'y':-45, 'z':0
        }
    }
    obj_coordinate = {
        'position':{
            'x': 0, 'y':1, 'z':0
        },
        'rotation':{
            'x': 0, 'y':0, 'z':0
        }
    }
    rotated = convert_world_to_agent_coordinate(obj_coordinate, agent_coordinate)
    eps = 0.01
    assert rotated['position']['x'] - (-2.1) < eps and rotated['position']['x'] - (1) < eps and rotated['position']['x'] - (-0.7) < eps

