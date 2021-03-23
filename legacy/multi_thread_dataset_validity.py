import argparse
import pdb
import threading

import ai2thor.controller
import ai2thor
import json
import sys, os

sys.path.append(os.path.abspath('../scripts'))
from plugins.ithor_arm_plugin.ithor_arm_constants import reset_environment_and_additional_commands, TRAIN_OBJECTS, TEST_OBJECTS, ENV_ARGS, transport_wrapper

from plugins.ithor_arm_plugin.arm_calculation_utils import initialize_arm, is_object_in_receptacle, is_agent_at_position

SCENES = ["FloorPlan{}_physics".format(str(i + 1)) for i in range(30)]

OBJECTS = TRAIN_OBJECTS # + TEST_OBJECTS#
# OBJECTS = TEST_OBJECTS


def test_initial_location(controller):
    for s in SCENES:
        reset_environment_and_additional_commands(controller, s)
        event1, event2, event3 = initialize_arm(controller)
        if not (event1.metadata['lastActionSuccess'] and event2.metadata['lastActionSuccess'] and event3.metadata['lastActionSuccess']):
            return False, 'failed for {}'.format(s)
    return True, ''

def test_train_data_points(controller):
    total_checked = 0
    total_error = 0
    total_reasons = {}
    for s in SCENES:
        for o in OBJECTS:
            print('Testing ', s, o)

            with open('datasets/ithor-armnav/valid_{}_positions_in_{}.json'.format(o, s)) as f:
                data_points = json.load(f)
            visible_data = [data for data in data_points[s] if data['visibility']]
            for datapoint in visible_data:
                result, message = check_datapoint_correctness(controller, datapoint)
                total_checked += 1
                if not result:
                    total_error += 1
                    for reason in message:

                        total_reasons.setdefault(reason, 0)
                        total_reasons[reason] += 1
                    # return False, message
                    print('total checked', total_checked, 'total error', total_error, 'reasons', total_reasons)

    return True, ''


def prune_data_points(controller):
    for s in SCENES:
        for o in OBJECTS:
            print('Pruning ', s, o)
            with open('datasets/ithor-armnav/valid_{}_positions_in_{}.json'.format(o, s)) as f:
                data_points = json.load(f)
            visible_data = [data for data in data_points[s] if data['visibility']]
            remaining_valid = []
            for ind, datapoint in enumerate(visible_data):
                if ind % 100 == 10:
                    print(ind, 'out of', len(visible_data))
                result, message = check_datapoint_correctness(controller, datapoint)
                if result:
                    remaining_valid.append(datapoint)
            print('out of ', len(visible_data), 'remained', len(remaining_valid))
            with open('datasets/ithor-armnav/pruned_v2_valid_{}_positions_in_{}.json'.format(o, s), 'w') as f:
                json.dump({s: remaining_valid}, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--prune', default=False, action='store_true')
    parser.add_argument('--parallel_thread', default=5, type=int)


    args = parser.parse_args()
    return args

def test_initial_location_thread(controller, thread_num):

    print('Testing initial location', thread_num)
    result, message = test_initial_location(controller)
    if result:
        print('Test Finished', thread_num,'Passed')
    else:
        print('Test Finished', thread_num,'Failed', message)

if __name__ == '__main__':
    args = parse_args()
    threads = []
    for index in range(args.parallel_thread):
        controller = ai2thor.controller.Controller(**ENV_ARGS)
        x = threading.Thread(target=test_initial_location_thread, args=(controller,index))
        threads.append(x)
    for x in threads:
        x.start()
    for index, thread in enumerate(threads):
        print("Main    : before joining thread %d.", index)
        thread.join()
        print("Main    : thread %d done", index)


