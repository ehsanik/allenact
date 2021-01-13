import argparse
import os
import pdb

list_of_servers = ['kiana-workstation', 'vision-server11', 'vision-server12', 'vision-server14', 'aws2', 'aws4', 'aws5', 'aws6', 'aws7', 'aws8', 'aws9', 'aws10', 'aws11']

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--servers', default=None)
    parser.add_argument('--sync_weights', default=False, action='store_true')
    parser.add_argument('--sync_specific_weight', default=None)
    parser.add_argument('--sync_back', default=False, action='store_true')


    args = parser.parse_args()
    if args.servers is None:
        args.servers = list_of_servers
    else:
        args.servers = [args.servers]
    return args

def main(args):

    for server in args.servers:
        print('syncing to ', server)
        command = 'rsync  -avz \
             --exclude .idea \
             --exclude __pycache__/ \
             --exclude runs/ \
             --exclude .DS_Store \
             --exclude .direnv \
             --exclude .envrc \
             --exclude .git \
             --exclude experiment_output/ \
             --exclude docs/ \
             --exclude trained_weights/  \
             --exclude pretrained_model_ckpts/  \
             --exclude trained_weights/do_not_sync_weights/  \
             ../allenact {}:~/'.format(server)
        if args.sync_weights:
            command = command.replace('--exclude trained_weights/ ', '')
        if args.sync_specific_weight is not None:
            print('Not implemented yet')
            pdb.set_trace()
            command = command.replace('--exclude trained_weights/ ', '')
        os.system(command)

def sync_back(args):
    for server in args.servers:
        print('syncing from ', server)
        command = 'rsync  -avz \
             --exclude ImageVisualizer/ \
             {}:~/allenact/experiment_output/visualizations ~/Desktop/server_results_sync/{}'.format(server, server)

        if args.sync_weights:
            command = command.replace('--exclude ImageVisualizer/ ', '')
        os.system(command)

if __name__ == '__main__':
    args = parse_args()
    if args.sync_back:
        sync_back(args)
    else:
        main(args)