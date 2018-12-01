# utility to manage remote computer in Google Cloud



instance= 'mini' # 'small'
remote_user = 'morigo2015'
local_cmd = ''

# instance= 'std1'
# remote_user = 'im'
# local_cmd = '--project cam-usa --zone us-central1-c '

gc_path = '/home/im/cloud/google-cloud-sdk/bin'
remote_path_root = f'{remote_user}@{instance}:/home/{remote_user}/mypy/cam_detect'
local_path_root = '/home/im/mypy/cam_detect'

#inst_name='im@small-1'

import os
import sys

commands = {
    'init': 'init from scratch (set software, create folders, send all necessary files',
    'send_all': 'send all (src,models) files to remote from local, and config file from /cloud',
    'send_src': 'send all source files from local to remote, not include cloud config file',
    'get_cfg': 'get cloud config file from remote, store at local computer in /cloud',
    'send_cfg': 'send cloud config file to remote',
    'get_output': 'get output.avi from remote to cloud/results',
    'get_evnt' : ' get event_images from remote' ,
    'send': 'send local files to respective folders on remote comp' ,
    'get': 'get remote files to respective folders in cloud/' ,
}

def print_info():
    print(f'defaults:')
    print(f'    remote instance = {remote_user}@{instance}')
    print(f'    remote path root = {remote_path_root}')
    print(f'    local path root  = {local_path_root}')
    if local_cmd:
        print(f'local cmd = {local_cmd}')

def print_help_info():
    print('usage:\npython3 util_cloud.py <command> [flag] [--recurse]\n\nwhere <command>:')
    for cmd in commands:
        print(f'{cmd:10s} : {commands[cmd]}')
    print('')
    print_info()

def main():

    try:
        if sys.argv[1] not in commands:
            print(f'illegal command: {sys.argv[1]}')
            raise KeyError
        func_name = f'do_{sys.argv[1]}()'
    except (KeyError, IndexError):
        print_help_info()
        exit(1)

    print_info()
    os.chdir(local_path_root)

    #print(f'func: {func_name}')
    eval(func_name)

def do_init():
    gc_cmd(f'scp {local_cmd} cloud/setup_cloud_computer.sh   {remote_user}@{instance}:/home/{remote_user}')
    gc_cmd(f'ssh {remote_user}@{instance} --command ./setup_cloud_computer.sh')
    do_send_all()

def do_send_all():
    gc_cmd(f'scp {local_cmd} production/models/*           {remote_path_root}/production/models')
    gc_cmd(f'scp {local_cmd} production/face_dataset/*.pkl {remote_path_root}/production/face_dataset')
    do_send_src()
    do_get_cfg()

def do_send_src():
    gc_cmd(f'scp {local_cmd} src/*.py    {remote_path_root}/src')

def do_get_cfg():
    gc_cmd(f'scp {local_cmd} {remote_path_root}/src/cam_detect_cfg_cloud.py  cloud/')

def do_send_cfg():
    gc_cmd(f'scp {local_cmd} cloud/cam_detect_cfg_cloud.py {remote_path_root}/src/' )

def do_get_output():
    os.makedirs('cloud/results',exist_ok=True)
    gc_cmd(f'scp {local_cmd} {remote_path_root}/production/videos/output.avi cloud/results/')

def do_get_evnt():
    os.makedirs('cloud/event_images',exist_ok=True)
    gc_cmd(f'scp {local_cmd} {remote_path_root}/production/event_images/* cloud/event_images/')

def do_send():
    try:
        loc_path = sys.argv[2]
    except (KeyError):
        print_help_info()
        exit(1)
    try:
        params = sys.argv[3]
    except IndexError:
        params = ''
    gc_cmd(f'scp {local_cmd} {params} {local_path_root}/{loc_path} {remote_path_root}')

def do_get():
    try:
        rem_path = sys.argv[2]
    except KeyError:
        print_help_info()
        exit(1)
    try:
        params = sys.argv[3]
    except IndexError:
        params = ''
    os.makedirs(f'{local_path_root}/cloud/{rem_path}',exist_ok=True)
    gc_cmd(f'scp {local_cmd} {params} {remote_path_root}/{rem_path} {local_path_root}/cloud/')

def gc_cmd(cmd):
    cmd_str = f'{gc_path}/gcloud compute {cmd}'
    print(f'$$$ gcloud compute {cmd}')
    r = os.system(cmd_str)
    if r != 0:
        print(f'error while execute system()={r}')

if __name__ == '__main__':
    main()