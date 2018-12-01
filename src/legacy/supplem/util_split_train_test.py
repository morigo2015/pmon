# for all files in boxed_folder - move according file from labelled_folder to train_folder
# files left in boxed_folder can be moved manually to some test_folder

import os
from cam_detect_cfg import cfg

moved_cnt={}
for label in cfg['face_labels_list']:
    label_folder = cfg['boxed_folder'] + label + '/'
    moved_cnt[label]=0
    if not os.path.exists(label_folder):
        print(f"no samples for label '{label}'")
        continue  # skip labels which have no samples

    for img_fname in os.listdir(label_folder):
        path_to_image = label_folder + img_fname

        from_folder = cfg['labelled_folder']+label+'/'
        to_folder   = cfg['train_folder']+label+'/'
        os.makedirs(to_folder,exist_ok=True)
        print    (f"mv '{from_folder+img_fname}' '{to_folder+img_fname}'")
        os.system(f"mv '{from_folder+img_fname}' '{to_folder+img_fname}'")
        moved_cnt[label] += 1

print(f" {moved_cnt} files moved from {cfg['labelled_folder']} to {cfg['train_folder']}, based on {cfg['boxed_folder']}")
for l in moved_cnt:
    print(f"label={l}  moved_cnt={moved_cnt[label]}")
print(f"totally moved {sum([moved_cnt[l] for l in moved_cnt])} files")