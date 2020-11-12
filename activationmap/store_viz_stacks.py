"""Activation map with preprocessing option"""

import os
import platform

from imageio import imread, imwrite
import numpy as np
from subprocess import check_output
import pandas as pd

from activationmap.grads import Grads
from activationmap.grad_ops import GradOps
import glob
import param_gedi as param
from memory_profiler import profile
import sys
from pympler import asizeof
import tensorflow as tf
from preprocessing.datagenerator import Dataspring

os_type = platform.system()
if os_type == 'Linux':
    prefix = '/mnt/finkbeinerlab'
if os_type == 'Darwin':
    prefix = '/Volumes/data'


def mem(obj, name):
    m = asizeof.asizeof(obj)
    print(name, m)


def batches_from_fold(source_fold, dead_fold, live_fold, batch_size, batch_num, parser=lambda x: x, ):
    # check_files = check_output(['find {}'.format(os.path.join(source_fold, '*.tif'))], shell=True).decode().split()
    check_files = glob.glob(os.path.join(source_fold, '*.tif'))

    files = []
    lbls = []
    for file in check_files:
        fn = file.split('/')[-1]
        subdir = file.split('/')[-2]
        cur_lbl = [0] * 2
        if batch_num == 'batch2':
            chk_dead = os.path.join(dead_fold, 'Images', subdir, fn)
            chk_live = os.path.join(live_fold, 'Images', subdir, fn)
        elif batch_num == 'batch1':
            ssdir = file.split('/')[-3]
            chk_dead = os.path.join(dead_fold, 'Images', ssdir, subdir, fn)
            chk_live = os.path.join(live_fold, 'Images', ssdir, subdir, fn)
        else:
            chk_dead = os.path.join(dead_fold, fn)
            chk_live = os.path.join(live_fold, fn)
        if os.path.exists(chk_dead):
            cur_lbl[0] += 1
        if os.path.exists(chk_live):
            cur_lbl[1] += 1

        if cur_lbl[0] != cur_lbl[1]:
            files.append(file)
            lbls.append(cur_lbl)
        else:
            print('cur lbl', cur_lbl)
            print('Couldn\'t find label for image at {}'.format(file))

    for i in range(0, len(files), batch_size):
        print('batch from fold 1')
        _files = files[i:i + batch_size]
        _names = list(map(lambda file: '.'.join(file.split('/')[-1].split('.')[:-1]), _files))
        _lbls = lbls[i:i + batch_size]
        _imgs = list(map(lambda file: parser(imread(file)), _files))
        print('batch from fold 2')

        yield tuple(map(np.array, (_imgs, _lbls, _names)))


def batches_from_ds(tfrecord):
    Dat = Dataspring(tfrecord)
    ds = Dat.datagen_base(istraining=False, count=1)
    for imgs, lbls, files in ds:
        imgs = imgs.numpy()
        lbls = lbls.numpy()
        files = files.numpy()
        fs = [f.decode() for f in files]
        names = np.array(['.'.join(file.split('/')[-1].split('.')[:-1]) for file in fs])
        yield (imgs, lbls, names)

# @profile
def batches_from_fold_no_labels(source_fold, dead_fold, live_fold, batch_size, parser=lambda x: x):
    # check_files = check_output(['find {}'.format(os.path.join(source_fold, '**', '*.tif'))], shell=True).decode().split()
    check_files = glob.glob(os.path.join(source_fold, '*.tif'))
    files = []
    lbls = []
    for file in check_files:
        # fn = file.split('/')[-1]
        cur_lbl = [0, 1]
        # if os.path.exists(os.path.join(dead_fold, fn)):
        #     cur_lbl[0] += 1
        # if os.path.exists(os.path.join(live_fold, fn)):
        #     cur_lbl[1] += 1

        if cur_lbl[0] != cur_lbl[1]:
            files.append(file)
            lbls.append(cur_lbl)
        else:
            print('Couldn\'t find label for image at {}'.format(file))

    for i in range(0, len(files), batch_size):
        _files = files[i:i + batch_size]
        _names = list(map(lambda file: '.'.join(file.split('/')[-1].split('.')[:-1]), _files))
        _lbls = lbls[i:i + batch_size]
        _imgs = list(map(lambda file: parser(imread(file)), _files))

        yield tuple(map(np.array, (_imgs, _lbls, _names)))


# @profile
def save_batch(g, imgs, lbls, base_path, conf_mat_paths, fnames=None, makepaths=True, layer_name='block5_conv3'):
    """
    Method computes the Guided GradCAM visualization of an image for both classes; saves these representations as a three-channel (image;correct_label_grad;wrong_label_grad) tif image

    Args:
        g: Grads object
        imgs: image data tensor
        lbls: array of one-hots
        base_path: directory within which to write
        conf_mat_paths: confusion matrix subpath names (first dimension corresponds to label, second corresponds to correctness of prediction)
        fnames: optional list of filenames to save with; defaults to integer index of each base image
        makepaths: whether to build the directory tree specified by base_path/conf_mat_paths
        layer_name: name of layer to differentiate wrt. for GradCAM

    Returns: list of the indices of images which were written

    """

    ggcam_gen = g.gen_ggcam_stacks(imgs, lbls, layer_name, ret_preds=True)  # grads.py
    writes = []
    res_dict = {'filename': [], 'label': [], 'prediction': []}
    for i, lbl in enumerate(lbls):
        g_stack, preds = next(ggcam_gen)

        path_stem = os.path.join(base_path, conf_mat_paths[np.argmax(lbl)][
            np.argmax(preds)])  # using confusion matrix path at [true label (0/1)][predicted correctly? (0/1)]
        filename = '{}.tif'.format(i if fnames is None else fnames[i])

        if makepaths:
            os.makedirs(path_stem, exist_ok=True)
        try:
            imwrite(os.path.join(path_stem, filename), g_stack)
            writes.append(i)
            if fnames is not None:
                res_dict['filename'].append(fnames[i])
                res_dict['label'].append(np.argmax(lbl))
                res_dict['prediction'].append(np.argmax(preds))
        except:
            print('Could not write image at index {}'.format(i))
    # mem(ggcam_gen, 'ggcam')
    # mem(g_stack, 'g_strack')
    # mem(writes, 'writes')
    # mem(res_dict, 'res_dict')
    # mem(imgs, 'imgs2')
    return res_dict


# @profile
def process_fold(g, source_fold, dead_fold, live_fold, dest_path, conf_mat_paths, batch_size=10, parser=lambda x: x,
                 layer_name='block5_conv3', has_labels=True, batch_num=None, tfrecord=None):
    pred_df = pd.DataFrame({'filename': [], 'label': [], 'prediction': []})
    if has_labels:
        # batch_gen = batches_from_fold(source_fold, dead_fold, live_fold, batch_size=batch_size,
        #                               batch_num=batch_num, parser=parser)
        batch_gen = batches_from_ds(tfrecord)
    else:
        batch_gen = batches_from_fold_no_labels(source_fold, dead_fold, live_fold, batch_size=batch_size, parser=parser)
    for imgs, lbls, names in batch_gen:
        print('save batch, {}'.format(names[0]))
        d = save_batch(g, imgs, lbls, dest_path, conf_mat_paths, fnames=names, makepaths=True, layer_name=layer_name)
        print('df')
        df = pd.DataFrame(d)
        if not has_labels:
            df.labels = -1
        print('pred df')

        pred_df = pd.concat((pred_df, df), ignore_index=True)
        # mem(batch_gen, 'batch_gen')
        # mem(pred_df, 'pred_df')
        # mem(imgs, 'imgs')
        # mem(lbls, 'lbls')
        # mem(d, 'd')
        # mem(g, 'g')

    print('to csv')

    pred_df.to_csv(os.path.join(dest_path, dest_path.split('/')[-1] + '.csv'))
    return 0


p = param.Param()

# timestamp = 'custom_2020_10_24_14_44_20'  # Sent (2020/10/24) Trained batch1, train batch2 is 82.9%, test batch2 79%, val batch2 77%
# timestamp = 'custom_2020_10_24_14_49_27'  # Sent (2020/10/24) Trained batch2, train batch1 82%, test batch1 82.1%, val batch1 81.8%

timestamp = 'custom_2020_10_30_16_50_53'  # preaug batch all

# import_path = "/mnt/data/MJFOX/saved_models/{}.h5".format(timestamp)  # custom_2020_10_24_14_49_27
import_path = '/mnt/data/MJFOX/saved_checkpoints/{}.hdf5'.format(timestamp) # for custom_2020_10_24_14_44_20
guidedbool = True
tfrecord = p.data_test

g = Grads(import_path, guidedbool=guidedbool)
gops = GradOps(vgg_normalize=True)

# experiment = 'batch1_slide_Cur_Curation_v4_2020-10-21-16-58-28'
if tfrecord is None:
    experiment = 'test_batch2_slide_Cur_Curation_v4_2020-10-21-16-58-28_cutoff_fn'
    # experiment = 'test_batch2_slide_Cur_Curation_v4_2020-10-21-11-45-02_cutoff_ablate.tfrecord'
else:
    experiment = tfrecord.split('/')[-1].split('.t')[0]
batch_num = 'custom'
# main_fold = f'/mnt/data/MJFOX/{experiment}'
main_fold = f'/mnt/data/MJFOX/gradcam_ims/{experiment}'
# source_fold_prefix = os.path.join(main_fold, 'batches')
source_fold_prefix = os.path.join(main_fold)

neg_fold = os.path.join(main_fold, 'negative')
pos_fold = os.path.join(main_fold, 'positive')

dest_path_prefix = prefix + f'/robodata/Josh/Gradcam/results/mjfox/{experiment}'
conf_mat_paths = [['artifact_true', 'artifact_false'], ['asyn_false', 'asyn_true']]
batch_size = 16
parser = lambda img: gops.img_parse(img)
# layer_name = 'block5_conv3'  # VGG16
# layer_name = 'block5_conv4'  # VGG19
layer_name = 'conv2d_4'  # custom_model
LABELLED = True
# layer_name = 'block1_conv1'

# # Example usage
# process_fold(g, source_fold, dead_fold, live_fold, dest_path, conf_mat_paths, batch_size=batch_size, parser=parser, layer_name=layer_name)
if batch_num == 'batch1':
    subdirs = glob.glob(os.path.join(main_fold, '**', 'Images', '**', 'S4-004_65_622187_SUBMANDIBULAR'))
    # subdirs = glob.glob(os.path.join(main_fold, '**'))
elif batch_num == 'batch2':
    subdirs = glob.glob(os.path.join(main_fold, '**', 'Images', 'S4-004_54_622186_SUBMANDIBULAR'))
else:
    subdirs = glob.glob(os.path.join(main_fold, '**'))

slidedir = [w.split('/')[-1] for w in subdirs]

# wells = [w for w in wells if w not in ['E5', 'B2', 'H10', 'B8', 'F7', 'H9', 'H3', 'C1', 'B10', 'E11', 'G4', 'F12', 'G12', 'D11', 'G9', 'G3', 'C10']]
# wells = ['HumanIncorrectDeadNoInnerSoma', 'HumanIncorrectLiveNoInnerSoma', 'HumanCorrectLiveInnerSoma']
# for well in map(str, range(3, 20 + 1)):
# for subdir in subdirs:
#     tifs = glob.glob(os.path.join(subdir, '*.tif'))
#     if len(tifs) >= batch_size:
#         print('Running {}'.format(subdir))
#         cur_source_fold = subdir
for _ in range(1):
    cur_source_fold = None
    # sl = subdir.split('/')[-1]
    # livedead = subdir.split('/')[-3]
    cur_dest_path = os.path.join(dest_path_prefix)

    process_fold(g, cur_source_fold, neg_fold, pos_fold, cur_dest_path, conf_mat_paths, batch_size=batch_size,
                 parser=parser, layer_name=layer_name, has_labels=LABELLED, batch_num=batch_num, tfrecord=tfrecord)
print(f'saved to {dest_path_prefix}')

