import copy

import matplotlib.pyplot as plt
import os
import json
import pdb
import ast
import numpy as np

base_dir = '/Users/kianae/Desktop/important_saved_models'
files ={
    # 'Blind Seen Objects Unseen Scenes': 'aws7/visualizations/BlindAllsceneALlobjOffWNavExp/11_13_2020_00_23_24/TestMetricLogger/test_metric_BlindAllsceneALlobjOffWNavExp.txt',
    #W nav
    # 'Val-SeenObj': 'kiana_workstation/visualizations/AllsceneAllobjOffWNavExp/11_12_2020_11_45_18/TestMetricLogger/test_metric_AllsceneAllobjOffWNavExp.txt',
    # 'Val-NovelObj': 'kiana_workstation/visualizations/UnseenObjUnseenSceneWNav/11_12_2020_11_46_20/TestMetricLogger/test_metric_UnseenObjUnseenSceneWNav.txt',
    # 'Train-NovelObj': 'kiana_workstation/visualizations/UnseenObjSeenSceneWNav/11_12_2020_11_46_39/TestMetricLogger/test_metric_UnseenObjSeenSceneWNav.txt',
    # 'Train-SeenObj': 'kiana_workstation/visualizations/SeenObjSeenSceneWNav/11_12_2020_11_47_00/TestMetricLogger/test_metric_SeenObjSeenSceneWNav.txt',
    #W-NAV
    'Test-SeenObj': 'full_saved_outputs/test_metric_RealDepthRandomAgentLocArmNav_03_24_2021_17_46_17_098605.txt',
    'Test-NovelObj': 'full_saved_outputs/test_metric_TestOnUObjUSceneRealDepthRandomAgentLocArmNav_03_24_2021_17_46_25_115330.txt',
    'SeenScenes-NovelObj': 'full_saved_outputs/test_metric_TestOnUObjSSceneRealDepthRandomAgentLocArmNav_03_24_2021_17_46_22_062785.txt',
    #No nav
    # 'Seen Objects Unseen Scenes': 'kiana_workstation/visualizations/11_11_2020_13_29_50/TestMetricLogger/test_metric.txt',
    # 'Unseen Objects Unseen Scenes': 'kiana_workstation/visualizations/11_11_2020_14_45_43/TestMetricLogger/test_metric.txt',
    # 'Unseen Objects Seen Scenes': 'kiana_workstation/visualizations/11_11_2020_14_48_51/TestMetricLogger/test_metric.txt',
    # 'Seen Objects Seen Scenes': 'kiana_workstation/visualizations/11_11_2020_14_52_33/TestMetricLogger/test_metric.txt',


}
files = {k:os.path.join(base_dir, f) for (k,f) in files.items()}

VALID_ACTIONS = ['MoveAheadContinuous', 'MoveArmHeightP', 'MoveArmZM', 'RotateLeftContinuous', 'MoveArmZP', 'RotateRightContinuous', 'MoveArmHeightM', 'MoveArmXM', 'MoveArmXP', 'MoveArmYP', 'MoveArmYM', 'Done', 'Pickup']
OBJECT_TYPES = ['Apple', 'Bread', 'Tomato', 'Lettuce', 'Pot', 'Mug', 'Potato', 'SoapBottle', 'Pan', 'Egg', 'Spatula', 'Cup']
COUNTERTOP_TYPES = ['Pot', 'Sink', 'Toaster', 'CounterTop', 'Plate', 'DiningTable', 'Chair', 'Bowl', 'SideTable', 'CoffeeMachine', 'StoveBurner', 'Cabinet', 'Mug', 'Pan', 'Fridge', 'Microwave', 'GarbageCan', 'Cup', 'Stool', 'Shelf', 'Drawer']

def process_file(f_name):
    with open(f_name) as f:
        data = [l for l in f]
    total_list = []
    for l in data:
        dict = ast.literal_eval(l.replace('\n', ''))
        total_list.append(dict)
    return total_list

def dist_two_location(p1, p2):
    p1 = p1['position']
    p2 = p2['position']
    return sum([(p1[k] - p2[k]) ** 2 for k in ['x','y','z']]) ** 0.5

def Sort_Tuple(tup):

    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    return(sorted(tup, key = lambda x: x[0]))

def show_plot(frequence, countertops, objects):

    fig, ax = plt.subplots()
    im = ax.imshow(frequence)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(countertops)))
    ax.set_yticks(np.arange(len(objects)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(countertops)
    ax.set_yticklabels(objects)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(objects)):
        for j in range(len(countertops)):
            text = ax.text(j, i, frequence[i, j],
                           ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()

def plot_files(file_names):
    total_results = {}

    all_countertop_types = []
    countertop_obj = {}
    color = ['#60ccb3', '#585ccc', '#8f52d1', '#de3ad8']
    all_distance_in_each_direction = {
        'x':[],
        'y':[],
        'z':[],
    }
    for exp_name in file_names:

        f=file_names[exp_name]
        data_point_list = process_file(f)
        total_results[f] = data_point_list
        distance_to_correctness = []
        distance_in_each_direction = {
            'x':[],
            'y':[],
            'z':[],
        }
        distance_to_actions = []

        for data_item in data_point_list:
            task_info = data_item['task_info_metrics']['task_info']
            target_location = task_info['target_location']
            source_location = dict(position=task_info['source_location']['object_location'])
            distance = dist_two_location(source_location, target_location)

            #plan 5
            distance_in_each_direction['x'] += [ abs(source_location['position']['x'] - target_location['position']['x'])]
            distance_in_each_direction['y'] += [ abs(source_location['position']['y'] - target_location['position']['y'])]
            distance_in_each_direction['z'] += [ abs(source_location['position']['z'] - target_location['position']['z'])]

            success = data_item['task_info_metrics']['success']
            success = 1 if success else 0

            #plan 1
            if success == 1:
                if not  data_item['task_info_metrics'][ 'metric/average/success_wo_disturb']:
                    success = 0


            counter_type = task_info['source_location']['countertop_id'].split('|')[0]
            distance_to_correctness.append((distance, success))
            object_type = task_info['objectId'].split('|')[0]
            all_countertop_types.append(counter_type)
            countertop_obj.setdefault((counter_type, object_type), 0)
            countertop_obj[(counter_type, object_type)] += 1

            action_list = data_item['action_sequence']
            distance_to_actions.append((distance, action_list))

        all_distance_in_each_direction = {
            k: all_distance_in_each_direction[k] + distance_in_each_direction[k]
            for k in distance_in_each_direction
        }
        # plt.scatter([d for (d,s) in distance_to_correctness], [s for (d,s) in distance_to_correctness])
        distance_to_correctness =Sort_Tuple(distance_to_correctness)
        all_success_list = []
        x_list = [d for (d,s) in distance_to_correctness]
        correctness= [s for (d,s) in distance_to_correctness]
        correctness = []
        #LATER_TODO
        for (d,s) in distance_to_correctness:
            if d > 4.5:
                correctness.append(0)
            else:
                correctness.append(s)
        for min_dist in range(len(x_list)):
            succ_rate = sum(correctness[min_dist:]) / len(correctness[min_dist:])
            all_success_list .append(succ_rate)

        #plan 9
        # distance_to_actions = Sort_Tuple(distance_to_actions)
        # all_actions_aggregated = []
        # def plot_stuff(success):
        #     x_distance = []
        #     label_translator = {
        #         ('MoveAheadContinuous',): 'Move Ahead',
        #         ('RotateLeftContinuous', 'RotateRightContinuous'): 'Rotate Agent',
        #         ('MoveArmHeightP', 'MoveArmHeightM'): 'Move Arm\'s Height',
        #         ('MoveArmXM', 'MoveArmXP', 'MoveArmYM', 'MoveArmYP', 'MoveArmZM', 'MoveArmZP'): 'Move Arm\'s Location'
        #     }
        #     so_far_aggregated = {v:0 for v in label_translator.keys()}
        #     for i in range(len(distance_to_actions)):
        #         (d, seq) = distance_to_actions[i]
        #         seq = [x for x in seq if x != 'MoveArmYM']
        #         if success is not None and distance_to_correctness[i][1] != success:
        #             continue # only correct ones
        #
        #         so_far_aggregated = copy.copy(so_far_aggregated)
        #         for action in seq:
        #             for k in so_far_aggregated:
        #                 if action in k:
        #                     so_far_aggregated[k] += 1
        #                     break
        #
        #         all_actions_aggregated.append((d, so_far_aggregated))
        #         x_distance += [d]
        #
        #     index_to_keep = [i for i in range(len(x_distance)) if x_distance[i] >= 0.15]
        #     for action_list in so_far_aggregated.keys():
        #         denom=1. / len(action_list)
        #         action_count = [denom * so_far_aggregated[action_list]/sum([v for v in so_far_aggregated.values()]) for (d,so_far_aggregated) in all_actions_aggregated]
        #         success_label = ''
        #         if success is not None:
        #             success_label = (' (Success)' if success else ' (Fail)')
        #         plt.plot(np.array(x_distance)[index_to_keep], np.array(action_count)[index_to_keep], label=label_translator[action_list] + success_label,linewidth=3)
        #
        # # plot_stuff(success=True)
        # # plot_stuff(success=False)
        # plot_stuff(success=None)
        #
        # plt.legend(loc="center right")
        # plt.ylabel('Percentage usage of an action')
        # plt.xlabel('Distance from Initial to Goal State (in m)')
        # plt.show()
        # pdb.set_trace()


        #plan 7
        # total_steps_needed = {
        #         k: [l/0.05 for l in distances]
        #         for (k,distances) in distance_in_each_direction.items()
        #     }
        # total_height_changing_steps = total_steps_needed['y']
        # total_horizontal_steps =[total_steps_needed['x'][i] + total_steps_needed['z'][i] for i in range(len(total_steps_needed['x']))]
        # hist, bin_edges = np.histogram(total_height_changing_steps, density=True)
        # mid_bins = [(bin_edges[i] + bin_edges[i - 1]) / 2 for i in range(1, len(bin_edges))]
        # plt.plot(mid_bins[0:], hist[0:], label=exp_name+' vertical steps needed')
        # hist, bin_edges = np.histogram(total_horizontal_steps, density=True)
        # mid_bins = [(bin_edges[i] + bin_edges[i - 1]) / 2 for i in range(1, len(bin_edges))]
        # plt.plot(mid_bins[0:], hist[0:], label=exp_name+' horizontal steps needed')
        # plt.legend(loc="upper right")
        # plt.xlabel('Estimate number of minimum steps')
        # plt.ylabel('Percentage of dataset')

        #plan 6
        # total_steps_needed = {
        #     k: [l/0.05 for l in distances]
        #     for (k,distances) in distance_in_each_direction.items()
        # }
        # total_distances = [total_steps_needed['x'][i] + total_steps_needed['y'][i] + total_steps_needed['z'][i] for i in range(len(total_steps_needed['x']))]
        # # plt.hist(total_distances, bins=20, label=exp_name, histtype='bar', weights=np.ones(len(total_distances)) / len(total_distances), color=color[0] + '50')
        # # color = color[1:]
        # hist, bin_edges = np.histogram(total_distances, density=True)
        # mid_bins = [(bin_edges[i] + bin_edges[i - 1]) / 2 for i in range(1, len(bin_edges))]
        # plt.plot(mid_bins[0:], hist[0:], label=exp_name)
        # plt.legend(loc="upper right")
        # plt.xlabel('Estimate number of steps')
        # plt.ylabel('Percentage of dataset')

        #plan 1
        # plt.plot(x_list, all_success_list, label=exp_name,linewidth=3)
        # plt.xlim(right=4.5)
        # plt.legend(loc="lower left")
        # plt.xlabel('Minimum distance between target and initial state (in m)')
        # plt.ylabel('Success without disturbance rate')
        # plt.grid(True)


        #plan 1.5
        plt.plot(x_list, all_success_list, label=exp_name,linewidth=3)
        plt.xlim(right=4.5)
        plt.legend(loc="lower left")
        plt.xlabel('Minimum distance between target and initial state (in m)')
        plt.ylabel('Success rate')
        plt.grid(True)

        #plan 2
        # plt.hist(x_list, bins='auto')

        #plan3
        # hist, bin_edges = np.histogram(x_list)
        # hist = [x / sum(hist) for x in hist]
        # mid_bins = [(bin_edges[i] + bin_edges[i - 1]) / 2 for i in range(1, len(bin_edges))]
        # plt.plot(mid_bins[1:], hist[1:], label=exp_name)
        # plt.legend(loc="upper right")
        # plt.xlabel('Distance from Initial to Goal State (in m)')
        # plt.ylabel('Percentage of dataset')



    #plan 4
    # all_countertop_types = list(set(all_countertop_types))
    # COUNTERTOP_TYPES = all_countertop_types
    # xs = []
    # ys = []
    # frequency_mat = np.zeros((len(COUNTERTOP_TYPES), len(OBJECT_TYPES)))
    # for (k, v) in countertop_obj.items():
    #     counter_ind = COUNTERTOP_TYPES.index(k[0])
    #     obj_ind = OBJECT_TYPES.index(k[1])
    #     xs += [counter_ind for _ in range(v)]
    #     ys += [obj_ind for _ in range(v)]
    #     frequency_mat[counter_ind, obj_ind] = v
    # # frequency_mat = frequency_mat / frequency_mat.sum(1).reshape(-1, 1)
    # # frequency_mat = (frequency_mat * 100).astype(int).astype(float) / 100
    # # np.histogram2d(xs, ys, bins=(len(COUNTERTOP_TYPES), len(OBJECT_TYPES)))
    # # plt.hist2d(xs, ys, bins=(len(COUNTERTOP_TYPES), len(OBJECT_TYPES)))
    # show_plot(frequency_mat, OBJECT_TYPES, COUNTERTOP_TYPES)


    # #plan 5
    # for axis in all_distance_in_each_direction:
    #     distances_this_axis = all_distance_in_each_direction[axis]
    #     distances_this_axis = [x / 0.05 for x in distances_this_axis if x <= 4]
    #     hist, bin_edges = np.histogram(distances_this_axis)
    #     hist = [k /sum(hist) for k in hist]
    #     mid_bins = [(bin_edges[i] + bin_edges[i - 1]) / 2 for i in range(1, len(bin_edges))]
    #     plt.plot(mid_bins[0:], hist[0:], label= 'Along axis ' + axis)
    #     # plt.hist(distances_this_axis, bins=20, label=axis, histtype='step', weights=np.ones(len(distances_this_axis)) / len(distances_this_axis))
    #     plt.legend(loc="upper right")
    #     plt.xlabel('Estimated minimum number of steps')
    #     plt.ylabel('Percentage')





    plt.show()
    pdb.set_trace()


plot_files(files)