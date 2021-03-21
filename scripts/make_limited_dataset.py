import copy
import json
import os
import pdb
import math

LEN_OF_CAP = 60

list_of_scenes = ['FloorPlan{}_physics'.format(str(i)) for i in range(1, 31)]
# list_of_objects = ['Potato', 'SoapBottle', 'Pan', 'Plate', 'Egg', 'Spatula', 'Cup', 'Bowl', 'SaltShaker', 'DishSponge'] #LATER_TODO do it for these later
list_of_objects = ['Apple', 'Bread', 'Tomato', 'Lettuce', 'Pot', 'Mug']
list_of_objects += ['Potato', 'SoapBottle', 'Pan', 'Egg', 'Spatula', 'Cup'] # add

BASE_DIRECTORY = 'datasets/ithor-armnav/'
PRUNED = 'pruned_v2_'

def calc_possible_trajectories(all_possible_points):


    countertop_object_to_data_id = {}
    # object_to_data_id = {}

    for i in range(len(all_possible_points)):
        countertop_id = all_possible_points[i]['countertop_id']
        object_id = all_possible_points[i]['object_id']
        counter_object = '{}_{}'.format(countertop_id, object_id)
        countertop_object_to_data_id.setdefault(counter_object, [])
        countertop_object_to_data_id[counter_object].append(i)

        # object_to_data_id.setdefault(object_id, [])
        # object_to_data_id[object_id].append(i)

    return countertop_object_to_data_id

def get_all_tuples_from_list(list):
    result = []
    for first_ind in range(len(list) - 1):
        for second_ind in range(first_ind + 1, len(list)):
            result.append([list[first_ind], list[second_ind]])
    return result


def generate_all_possible_sequences(countertop_object_to_data_id):
    result = []
    for (k, v) in countertop_object_to_data_id.items():
        if len(v) <= 1:
            continue
        result += get_all_tuples_from_list(v)

    return result

def load_all_possible_positions(obj, scene):
    all_possible_points = []
    original_json_file = os.path.join(BASE_DIRECTORY, PRUNED + 'valid_{}_positions_in_{}.json'.format(obj, scene))
    with open(original_json_file) as f:
        data_points = json.load(f)
    visible_data = [data for data in data_points[scene] if data['visibility']]
    all_possible_points += visible_data
    return all_possible_points



def get_sample_w_nav_sequences(countertop_object_to_data_id, sample_size, dataset):
    MIN_VALID = 3 #
    # this only includes
    # {'StoveBurner', 'Stool', 'Shelf', 'Chair', 'DiningTable', 'Sink', 'CounterTop', 'Cabinet'} # IMPORTANT
    original = copy.copy(countertop_object_to_data_id)
    valid_countertops = [k for (k,v) in countertop_object_to_data_id.items() if len(v) > MIN_VALID]

    if len(valid_countertops) == 0:
        print('Oh Shoot did not WORK OUT')
        return []

    import random
    random.seed(0)

    result = []
    for i in range(sample_size):
        first_counter = random.choice(valid_countertops)
        second_counter = random.choice(valid_countertops)
        data_point_ind_0 = random.choice(countertop_object_to_data_id[first_counter])
        data_point_ind_1 = random.choice(countertop_object_to_data_id[second_counter])
        data_point_0 = dataset[data_point_ind_0]
        data_point_1 = dataset[data_point_ind_1]

        assert data_point_0['object_id'] == data_point_1['object_id']
        # assert data_point_0['countertop_id'] == data_point_1['countertop_id']
        assert data_point_0['scene_name'] == data_point_1['scene_name']
        result.append((data_point_0, data_point_1))

    random.shuffle(result)
    # [v[0]['countertop_id'] for v in result]
    return result

def main_w_nav():

    for scene in list_of_scenes:
        for obj in list_of_objects:

            print('obj, scene', obj, scene)
            all_possible_points = load_all_possible_positions(obj, scene)

            countertop_object_to_data_id = calc_possible_trajectories(all_possible_points)

            result = get_sample_w_nav_sequences(countertop_object_to_data_id, LEN_OF_CAP, all_possible_points)

            print(len(result))

            output_file = os.path.join(BASE_DIRECTORY, PRUNED + 'w_nav_tasks_{}_positions_in_{}.json'.format(obj, scene))
            with open(output_file, 'w') as f:
                json.dump({
                    scene: result
                }, f)

def position_distance(pos1, pos2):
    return sum([(pos1[k] - pos2[k]) ** 2 for k in ['x', 'y', 'z']]) ** .5

def hard_get_sample_w_nav_sequences(countertop_object_to_data_id, sample_size, dataset):
    MIN_VALID = 2
    MINIMUM_GOAL_START_DISTANCE = 0.5
    # this only includes
    # {'StoveBurner', 'Stool', 'Shelf', 'Chair', 'DiningTable', 'Sink', 'CounterTop', 'Cabinet'} # IMPORTANT
    original = copy.copy(countertop_object_to_data_id)
    valid_countertops = [k for (k,v) in countertop_object_to_data_id.items() if len(v) > MIN_VALID]

    if len(valid_countertops) == 0:
        print('Oh Shoot did not WORK OUT')
        return []

    import random
    random.seed(0)

    result = []
    result_dict = []
    while(len(result) < sample_size):

        first_counter = random.choice(valid_countertops)
        remainder_countertops = copy.deepcopy(valid_countertops)
        random.shuffle(remainder_countertops)
        for second_counter in remainder_countertops:
            if first_counter == second_counter:
                continue
            data_point_ind_0 = random.choice(countertop_object_to_data_id[first_counter])
            data_point_ind_1 = random.choice(countertop_object_to_data_id[second_counter])


            data_point_0 = dataset[data_point_ind_0]
            data_point_1 = dataset[data_point_ind_1]

            init_obj_location = data_point_0['object_location']
            goal_obj_location = data_point_1['object_location']
            if position_distance(init_obj_location, goal_obj_location) < MINIMUM_GOAL_START_DISTANCE:
                continue

            assert data_point_0['object_id'] == data_point_1['object_id']
            # assert data_point_0['countertop_id'] == data_point_1['countertop_id']
            assert data_point_0['scene_name'] == data_point_1['scene_name']

            if (data_point_0, data_point_1) in result_dict: #to make sure we don't add repeated elements
                continue
            result_dict.append((data_point_0, data_point_1))
            result.append((data_point_0, data_point_1))
            break

    random.shuffle(result)
    # are we covering many countertops or what?
    # [v[0]['countertop_id'] for v in result]
    return result

def hard_main_w_nav():

    for scene in list_of_scenes:
        for obj in list_of_objects:

            print('obj, scene', obj, scene)
            all_possible_points = load_all_possible_positions(obj, scene)

            countertop_object_to_data_id = calc_possible_trajectories(all_possible_points)

            result = hard_get_sample_w_nav_sequences(countertop_object_to_data_id, LEN_OF_CAP, all_possible_points)

            print(len(result))

            output_file = os.path.join(BASE_DIRECTORY, PRUNED + 'hard_w_nav_tasks_{}_positions_in_{}.json'.format(obj, scene))
            with open(output_file, 'w') as f:
                json.dump({
                    scene: result
                }, f)


if __name__ == '__main__':
    # main_no_nav()
    # main_w_nav()
    hard_main_w_nav()








# def get_sample_sequences(countertop_object_to_data_id, sample_size, dataset):
#     MIN_VALID = 4 TODO
#     # this only includes
#     # {'StoveBurner', 'Stool', 'Shelf', 'Chair', 'DiningTable', 'Sink', 'CounterTop', 'Cabinet'} # IMPORTANT
#     original = copy.copy(countertop_object_to_data_id)
#
#     total_count = [len(v) * (len(v) -1 ) / 2. for v in countertop_object_to_data_id.values() if len(v) > MIN_VALID]
#     if sum(total_count) == 0:
#         print('Oh Shoot did not WORK OUT')
#         return []
#     cap_training = math.sqrt(sample_size / sum(total_count))
#     if cap_training > 1: # which means total_count is smaller than sample size
#         cap_training = 1
#     import random
#     random.seed(0)
#     for countertop_obj in countertop_object_to_data_id.keys():
#         all_sequence = countertop_object_to_data_id[countertop_obj]
#         count_to_keep = math.ceil(len(all_sequence) * cap_training)
#         if len(all_sequence) <= MIN_VALID:
#             continue
#         # if count_to_keep <= 2:
#         #     count_to_keep = len(all_sequence)
#         count_to_keep = max(count_to_keep, 2)
#         all_sequence = random.sample(all_sequence, count_to_keep)
#         countertop_object_to_data_id[countertop_obj] = all_sequence
#
#     result = []
#     for same_counters in countertop_object_to_data_id.values():
#         for i in range(len(same_counters) - 1):
#             for j in range(i + 1, len(same_counters)):
#                 data_point_0 = dataset[same_counters[i]]
#                 data_point_1 = dataset[same_counters[j]]
#                 assert data_point_0['object_id'] == data_point_1['object_id']
#                 assert data_point_0['countertop_id'] == data_point_1['countertop_id']
#                 assert data_point_0['scene_name'] == data_point_1['scene_name']
#                 result.append((data_point_0, data_point_1))
#
#     random.shuffle(result)
#     # [v[0]['countertop_id'] for v in result]
#     return result

# def main_no_nav():
#
#     for scene in list_of_scenes:
#         for obj in list_of_objects:
#
#             print('obj, scene', obj, scene)
#             all_possible_points = load_all_possible_positions(obj, scene)
#
#             countertop_object_to_data_id = calc_possible_trajectories(all_possible_points)
#
#             result = get_sample_sequences(countertop_object_to_data_id, LEN_OF_CAP, all_possible_points)
#
#             print(len(result))
#
#             output_file = os.path.join(BASE_DIRECTORY, PRUNED + 'no_nav_tasks_{}_positions_in_{}.json'.format(obj, scene))
#             with open(output_file, 'w') as f:
#                 json.dump({
#                     scene: result
#                 }, f)
#

