import argparse
import json
import pdb
import random


def parse_args():
    parser = argparse.ArgumentParser(description='analyze')
    parser.add_argument('--max_per_scene', default=1000, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open('datasets/ithor-armnav/valid_agent_initial_locations.json') as f:
        valid_points = json.load(f)

    deterministic_test_dict = {}
    for scene in valid_points:
        full_list = valid_points[scene]
        deterministic_list = [random.choice(full_list) for _ in range(args.max_per_scene)]
        deterministic_test_dict[scene] = deterministic_list

    print('are you sure you want to rewrite?')
    pdb.set_trace()

    with open('datasets/ithor-armnav/deterministic_valid_agent_initial_locations.json', 'w') as f:
        json.dump(deterministic_test_dict, f)
