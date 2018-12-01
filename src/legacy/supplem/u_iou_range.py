# for clips in folder print IoU range and some other stats

import argparse, os, glob
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()

def iou_lst(folder):
    return [ int(f.split(sep='/')[-1].split(sep='_')[2]) for f in glob.glob(f'{folder}/*.avi') ]

pers = iou_lst(f'{args.folder}/persons')
noper = iou_lst(f'{args.folder}/noperson')
print(f'{args.folder}: ')
print(f' Total:    cnt={len(pers+noper):4}   {min(pers+noper):3} - {max(pers+noper):3}')
print(f' persons:  cnt={len(pers):4}   {min(pers):3} - {max(pers):3}')
print(f' noperson: cnt={len(noper):4}   {min(noper):3} - {max(noper):3}')

