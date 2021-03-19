import argparse
import json
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='analyze')
    parser.add_argument('--metric_file', default='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.metric_file) as f:
        result_list = json.load(f)

    assert len(result_list) == 1
    result_list = result_list[0]

    success = result_list['success']
    pickup_succ = result_list['metric/average/final_obj_pickup']
    eplen_success = result_list['metric/average/eplen_success']
    eplen_pickup = result_list['metric/average/eplen_pickup']
    print(dict(success=success, pickup_succ=pickup_succ, eplen_success=eplen_success, eplen_pickup=eplen_pickup))

    tasks = result_list['tasks']

    #just a sanity check
    average_success = []
    average_eplen = []
    for t in tasks:
        average_success.append(t['success'])
        if t['success']:
            average_eplen.append(t['metric/average/eplen_success'])
    print('average success', sum(average_success) / len(average_success))
    print('average eplen success', sum(average_eplen) / len(average_eplen))

    pdb.set_trace()