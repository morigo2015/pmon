# util to cluster entry/exit points based on cilps filenames info

# actions:
# 1) build/add path_info array, based on videos/ clips info (start box, end box, enclosing box, cnt, interv)
# 2) path_info array --> entries classifier
# 3) path_info array, entries --> visualize

input = {'1' : '/home/im/mypy/cam_detect/preparation/clust/in/', '2':'/home/im/mypy/cam_detect/preparation/clust/out/'}
output_folder = '/home/im/mypy/cam_detect/preparation/clust/analyze/'
bg_file =       '/home/im/mypy/cam_detect/preparation/clust/cam_background.png'
input_mode = '1'

import os
import glob
import numpy as np
import cv2

from cam_boxes import Box, COLORS,BGR_GRAY

def main():
    input_data = build_input_data(None, get_filenames(input['1']), mode='1')
    input_data = build_input_data(input_data, get_filenames(input['2']), mode='2')

    #try_dbscan(input_data)
    try_agglomer(input_data)

def get_filenames(folder):

    # read *.lst and list of *.avi files; convert to ndarray
    filenames = []
    file_cnt = 0
    for f in glob.glob(folder+'*.avi'):
        #print(f'avi file: {f}')
        filenames.append(f)
        file_cnt += 1
    for f in glob.glob(folder+'*.lst'):
        #print(f'lst file: {f}')
        for l in open(f,'r').readlines():
            filenames.append(l)
            file_cnt += 1
    print(f'Read {file_cnt} filenames.')
    return  filenames

def build_input_data(old_data, filenames_lst, mode):
    # mode: 1 - take first box only, 2 - second only, 12 - first and second
    if old_data is None:
        data = np.empty((0,4),dtype=int)
    else:
        data = old_data

    for fname in filenames_lst:
        fname_noext = fname.split(sep='.')[-2]
        fields = fname_noext.split(sep='_')
        start_box = fields[-2]
        end_box = fields[-1]
        if '1' in mode:
            data = np.vstack( (data,Box.str_2_coord(start_box)) )
        if '2' in mode:
            data = np.vstack( (data,Box.str_2_coord(end_box)) )
    print(f'data shape={data.shape}')
    return data

# --------- Agglomerative -------------------------------------------------------------------------------------

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def try_agglomer(input_data):
    for n_clusters in range(1,2):
        #clust_labels = [f'clust{n}' for n in range(n_clusters)]
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        agg.fit(input_data)
        print(f'\nAgg: n_clust = {n_clusters} leaves={agg.n_leaves_} components={agg.n_components_} ')

        print_clust_stat(agg,'agg')
        draw_frames(agg,input_data,n_clusters,'agg')
        draw_centers(agg, input_data, n_clusters,'agg')
        draw_IOU(agg, input_data, n_clusters,'agg')

# --------- DBSCAN --------------------------------------------------------------------------------------------
from sklearn.cluster import DBSCAN

def try_dbscan(input_data):
    for param in np.arange(0.05,0.7,0.1): #    range(20,90,10):
        l_size = int(param*len(input_data))
        dbscan = DBSCAN(leaf_size=l_size)  # eps=eps)
        dbscan.fit(input_data)
        uniq, cnt = np.unique(dbscan.labels_, return_counts=True)
        print(f'\nDBSCAN: param={param:4}, labels:{len(set(dbscan.labels_)):3d},    lab:cnt  {dict(zip(uniq,cnt))}')

        print_clust_stat(dbscan,'dbscan')
        draw_frames(dbscan,input_data,param, 'dbscan')
        draw_centers(dbscan,input_data,param, 'dbscan')
        draw_IOU(dbscan,input_data,param, 'dbscan')

# -----------------------------------

def print_clust_stat(model,name):
    labels = model.labels_
    n_clust = len(set(labels))
    msg = f'method = {name} '
    msg += f'Total amount of items = {len(labels)} '
    msg += f'number of clusters = {n_clust}\n'
    for n in range(n_clust):
        cnt = sum(labels==n)
        msg += f'cluster {n}: {cnt} ({cnt*100./len(labels):4.0f}%)\n'
    print(msg)

def draw_frames(dbs, data, param, name):
    img = cv2.imread(bg_file)
    colors = [c for c in COLORS if c!=BGR_GRAY]
    for ind,b in enumerate(data):
        corn_tuple = tuple(b)
        box = Box(corners_tuple=corn_tuple)
        clust_num = dbs.labels_[ind]
        c = BGR_GRAY if clust_num == -1 else colors[clust_num]
        box.draw(img,color=c)
    cv2.imwrite(output_folder +f'{name}_'+'frames_' + f'param:{param:04.0f}' + '.jpg', img)

def draw_centers(dbs, data, param,name):
    img = cv2.imread(bg_file)
    colors = [c for c in COLORS if c!=BGR_GRAY]
    for ind,b in enumerate(data):
        corn_tuple = tuple(b)
        box = Box(corners_tuple=corn_tuple)
        clust_num = dbs.labels_[ind]
        c = BGR_GRAY if clust_num == -1 else colors[clust_num]
        box.draw_center(img,color=c, size=10)
    cv2.imwrite(output_folder +f'{name}_'+'centers_' + f'param:{param:04.0f}' + '.jpg', img)

def draw_IOU(dbs, data, param,name):
    img = cv2.imread(bg_file)
    colors = [c for c in COLORS if c!=BGR_GRAY]
    i_box = {}
    u_box = {}
    cnt = {}
    for ind,b in enumerate(data):
        box = Box(corners_tuple=tuple(b))
        clust_num = dbs.labels_[ind]
        if clust_num in i_box:
            i_box[clust_num].intersect(box)
            u_box[clust_num].union(box)
            cnt[clust_num] += 1
        else:
            i_box[clust_num] = Box(corners_tuple=tuple(b))
            u_box[clust_num] = Box(corners_tuple=tuple(b))
            cnt[clust_num] = 1

    for clust_num in sorted(i_box.keys()):
        c = BGR_GRAY if clust_num == -1 else colors[clust_num]
        i_box[clust_num].draw2(img,color=c,label=f'I {clust_num} {cnt[clust_num]}({cnt[clust_num]*100./len(data):4.0f}%) ',thickness=3)
        u_box[clust_num].draw2(img,color=c,label=f'        {clust_num}(uni)',thickness=2)
        print(f'clust:={clust_num}  intr:{i_box[clust_num].repr()}  union:{u_box[clust_num].repr()}')
    cv2.imwrite(output_folder +f'{name}_'+'iou_' + f'param:{param:04.0f}' + '.jpg', img)


if __name__ == '__main__':
    main()