import glob
import os

def command(cmd_str):
    print('exec: ', cmd_str)
    os.system(cmd_str)

os.chdir('/home/im/mypy/cam_detect/event_images')
print(os.getcwd())
for d in ['2018-07-29','2018-07-28', '2018-07-27','2018-07-30', ]:
    cmd_str = 'mkdir {}'.format(d)
    command(cmd_str)

    lst = glob.glob(d+'*')
    #print ('d=',d, 'lst=',lst)
    for f in lst:
        cmd_str = 'mv "{}" {}'.format(f,d)
        command(cmd_str)