import argparse
import pdb

import ai2thor.controller
import ai2thor
import json
import sys, os

sys.path.append(os.path.abspath('.'))
from plugins.ithor_arm_plugin.ithor_arm_constants import reset_environment_and_additional_commands, TRAIN_OBJECTS, TEST_OBJECTS, ENV_ARGS, transport_wrapper

from plugins.ithor_arm_plugin.arm_calculation_utils import initialize_arm

SCENES = ["FloorPlan{}_physics".format(str(i + 1)) for i in range(30)]

OBJECTS = TRAIN_OBJECTS # TODO for now + TEST_OBJECTS

PRUNING = True


def test_initial_location(controller):
    for s in SCENES:
        reset_environment_and_additional_commands(controller, s)
        event1, event2, event3 = initialize_arm(controller)
        if not (event1.metadata['lastActionSuccess'] and event2.metadata['lastActionSuccess'] and event3.metadata['lastActionSuccess']):
            return False, 'failed for {}'.format(s)
    return True, ''

def check_datapoint_correctness(controller, source_location):
    scene = source_location['scene_name']
    reset_environment_and_additional_commands(controller, scene)
    event_place_obj = transport_wrapper(controller, source_location['object_id'], source_location['object_location'])
    _1, _2, _3 = initialize_arm(controller) #This is checked before
    agent_state = source_location['agent_pose']
    event_TeleportFull = controller.step(dict(action='TeleportFull', standing=True, x=agent_state['position']['x'], y=agent_state['position']['y'], z=agent_state['position']['z'], rotation=dict(x=agent_state['rotation']['x'], y=agent_state['rotation']['y'], z=agent_state['rotation']['z']), horizon=agent_state['cameraHorizon']))

    object_id = source_location['object_id']
    object_state = [o for o in event_TeleportFull.metadata['objects'] if o['objectId'] == object_id][0]
    object_is_visible = object_state['visible']
    # check to transport object

    # check to do arm init
    # check to transport agent
    # check object is visible
    if event_place_obj.metadata['lastActionSuccess'] and event_TeleportFull.metadata['lastActionSuccess'] and object_is_visible:
        return True, ''
    else:
        return False, 'Data point invalid for {}, because of event_place_obj {}, event_TeleportFull{}, object_is_visible {}'.format(source_location, event_place_obj, event_TeleportFull, object_is_visible)

def test_train_data_points(controller):
    for s in SCENES:
        for o in OBJECTS:
            print('Testing ', s, o)

            with open('datasets/ithor-armnav/pruned_valid_{}_positions_in_{}.json'.format(o, s)) as f:
                data_points = json.load(f)
            visible_data = [data for data in data_points[s] if data['visibility']]
            for datapoint in visible_data:
                result, message = check_datapoint_correctness(controller, datapoint)
                if not result:
                    return False, message

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
            with open('datasets/ithor-armnav/pruned_valid_{}_positions_in_{}.json'.format(o, s), 'w') as f:
                json.dump({s: remaining_valid}, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--prune', default=False, action='store_true')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    controller = ai2thor.controller.Controller(
        **ENV_ARGS
        # gridSize=0.25,
        # width=224, height=224, agentMode='arm', fieldOfView=100,
        # agentControllerType='mid-level',
        # server_class=ai2thor.fifo_server.FifoServer, visibilityScheme='Distance',
        # useMassThreshold = True, massThreshold = 10,
        # visibilityDistance = 1.25
    )

    print('Testing initial location')
    result, message = test_initial_location(controller)
    if result:
        print('Passed')
    else:
        print('Failed', message)

    if args.prune:
        print('Are you sure?')
        prune_data_points(controller)
    else:

        print('Testing test_train_data_points')
        result, message = test_train_data_points(controller)
        if result:
            print('Passed')
        else:
            print('Failed', message)
