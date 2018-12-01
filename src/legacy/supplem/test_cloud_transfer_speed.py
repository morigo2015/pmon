import os
import time
import subprocess
import types
import sys

cur_dir = '/home/im/mypy/'
inp_dir = 'tst2'
instance_name = 'mini'   # 'cpu1-std-2'
num_iter = 5


def dir_size(dir:str) -> int:
    cmd = f'du -s {inp_dir} '
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output,err) = p.communicate()
    return int(output.decode("utf-8").split()[0])

os.chdir(cur_dir)
print(f'cur_dir={os.getcwd()}  dir={inp_dir} instance={instance_name}' )

size_mb = dir_size(inp_dir) / 1024
print(f'size of {inp_dir} = {size_mb:8.3f}mb')

for i in range(num_iter):

    start = time.time()

    cmd_cp = f'gcloud compute scp --recurse {inp_dir}  morigo2015@{instance_name}:rem_tst'
    cmd = f'export PATH=$PATH:/home/im/cloud/google-cloud-sdk/bin\n{cmd_cp}'
    print(f'cmd={cmd}')

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here

    #p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    #(output, err) = p.communicate()

    dtime = time.time()-start

    print(f'---iter {i}   time = {dtime:8.2f}  speed = {size_mb/dtime:10.2f}-----------------------------')

    try:
        total_time += dtime
    except:
        total_time = dtime

print(f'{time.asctime()}: size={num_iter*size_mb:8.2f}mb  dt={dtime:6.2f}s   speed = {num_iter*size_mb/total_time:10.2f}mb/s')
#print(f'\n\noutput:\nlen={len(output)}\n{output.decode()}')