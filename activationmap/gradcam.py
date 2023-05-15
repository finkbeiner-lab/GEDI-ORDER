"""
Activation map with preprocessing option
Set main_fold to None to use tfrecord, set deploy_tfrec to None to run images in gradcam.
"""

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
# from memory_profiler import profile
import sys
import argparse
import pyfiglet
from pympler import asizeof
import tensorflow as tf
from preprocessing.datagenerator import Dataspring

__author__ = 'Josh Lamstein'


def mem(obj, name):
    m = asizeof.asizeof(obj)
    print(name, m)


def batches_from_fold(source_fold, dead_fold, live_fold, batch_size, parser=lambda x: x, ):
    # check_files = check_output(['find {}'.format(os.path.join(source_fold, '*.tif'))], shell=True).decode().split()
    check_files = glob.glob(os.path.join(source_fold, '*.tif'))

    files = []
    lbls = []
    for file in check_files:
        fn = file.split('/')[-1]
        subdir = file.split('/')[-2]
        cur_lbl = [0] * 2

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


def batches_from_ds(p, tfrecord):
    Dat = Dataspring(p, tfrecord)
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
def save_batch(g, imgs, lbls, base_path, conf_mat_paths, fnames=None, makepaths=True, layer_name='block5_conv3',
               gray_morphology_bool=True):
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

    ggcam_gen = g.gen_ggcam_stacks(imgs, lbls, layer_name, ret_preds=True,
                                   gray_morphology=gray_morphology_bool)  # grads.py
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
def process_fold(p, g, source_fold, dead_fold, live_fold, dest_path, conf_mat_paths, batch_size=10, parser=lambda x: x,
                 layer_name='block5_conv3', gray_morphology_bool=True, has_labels=True, tfrecord=None):
    pred_df = pd.DataFrame({'filename': [], 'label': [], 'prediction': []})
    if has_labels:
        if source_fold is not None:
            batch_gen = batches_from_fold(source_fold, dead_fold, live_fold, batch_size=batch_size, parser=parser)
        else:
            batch_gen = batches_from_ds(p, tfrecord)
    else:
        batch_gen = batches_from_fold_no_labels(source_fold, dead_fold, live_fold, batch_size=batch_size, parser=parser)
    for imgs, lbls, names in batch_gen:
        print('save batch, {}'.format(names[0]))
        d = save_batch(g, imgs, lbls, dest_path, conf_mat_paths, fnames=names, makepaths=True, layer_name=layer_name,
                       gray_morphology_bool=gray_morphology_bool)
        # print('df')
        df = pd.DataFrame(d)
        if not has_labels:
            df.labels = -1
        # print('pred df')

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


def run_gradcam(main_fold, dest_fold, deploy_tfrec, model_path, batch_size, layer_name='block5_conv3',
                gray_morphology_bool=True, imtype='tif'):
    p = param.Param(parent_dir=dest_fold, res_dir=dest_fold)
    if main_fold is not None and deploy_tfrec is not None:
        assert 0, 'main fold or deploy_tfrec must be Nan valued (None).'
    guidedbool = True

    g = Grads(model_path, guidedbool=guidedbool)
    gops = GradOps(p, vgg_normalize=True)

    pos_fold, neg_fold = None, None

    # conf_mat_paths = [['artifact_true', 'artifact_false'], ['asyn_false', 'asyn_true']]
    conf_mat_paths = [['zero_true', 'zero_false'], ['one_false', 'one_true']]
    parser = lambda img: gops.img_parse(img)
    # layer_name = 'block5_conv3'  # VGG16
    # layer_name = 'block5_conv4'  # VGG19
    # layer_name = 'conv2d_4'  # custom_model
    LABELLED = True if main_fold is None else False
    # layer_name = 'block1_conv1'
    if main_fold is not None:
        subdirs = glob.glob(os.path.join(main_fold, '**'))
        if os.path.isdir(subdirs[0]):
            for subdir in subdirs:
                tifs = glob.glob(os.path.join(subdir, f'*.{imtype}'))
                if len(tifs) >= batch_size:
                    print('Running {}'.format(subdir))
                    name = subdir.split('/')[-1]
                    cur_source_fold = subdir
                    cur_dest_path = os.path.join(dest_fold, name)
                    process_fold(p, g, cur_source_fold, neg_fold, pos_fold, cur_dest_path, conf_mat_paths,
                                 batch_size=batch_size,
                                 parser=parser, layer_name=layer_name, gray_morphology_bool=gray_morphology_bool,
                                 has_labels=LABELLED,
                                 tfrecord=deploy_tfrec)
                    print(f'saved to {cur_dest_path}')
        # Directory does not have subdirectories
        else:
            print(f'No subdirectories found, looking for images im {main_fold}')
            cur_source_fold = main_fold
            # sl = subdir.split('/')[-1]
            # livedead = subdir.split('/')[-3]
            cur_dest_path = dest_fold

            process_fold(p, g, cur_source_fold, neg_fold, pos_fold, cur_dest_path, conf_mat_paths,
                         batch_size=batch_size,
                         parser=parser, layer_name=layer_name, gray_morphology_bool=gray_morphology_bool,
                         has_labels=LABELLED, tfrecord=deploy_tfrec)
    else:
        print(f'Running gradcam on tfrecord: {deploy_tfrec}')
        cur_source_fold = None

        process_fold(p, g, cur_source_fold, neg_fold, pos_fold, dest_fold, conf_mat_paths, batch_size=batch_size,
                     parser=parser, layer_name=layer_name, gray_morphology_bool=gray_morphology_bool,
                     has_labels=LABELLED,
                     tfrecord=deploy_tfrec)
    print(f'saved to {dest_fold}')


if __name__ == '__main__':
    result = pyfiglet.figlet_format("GEDI-CNN Gradcam", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Gradcam on GEDICNN model')
    parser.add_argument('--im_dir', action="store",
                        # default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/Foxo1_Trap1/GXYTMP/17AAG_R5_IXM/CroppedImages/A09',
                        default=None,
                        help='directory of images to run', dest="im_dir")
    parser.add_argument('--model_path', action="store",
                        # default='/gladstone/finkbeiner/linsley/GEDI_CLUSTER/gedicnn.h5',
                        default='/gladstone/finkbeiner/lab/MITOPHAGY/saved_models/vgg19_2023_05_13_12_33_47.h5',
                        help='path to h5 or hdf5 model', dest="model_path")
    parser.add_argument('--deploy_tfrec', action="store", default='/gladstone/finkbeiner/lab/MITOPHAGY/test.tfrecord',
                        help='results directory', dest="deploy_tfrec")
    parser.add_argument('--layer_name', action="store", default='block5_conv4',
                        help='visualize layer', dest="layer_name")
                        # layer_name = 'block5_conv3'  # VGG16
                        # layer_name = 'block5_conv4'  # VGG19
                        # layer_name = 'conv2d_4'  # custom_model
    parser.add_argument('--resdir', action="store",
                        default='/gladstone/finkbeiner/lab/MITOPHAGY/gradcam',
                        help='results directory', dest="resdir")
    parser.add_argument('--batch_size', type=int, action="store", default=16,
                        help='Batch size. The remainder of total images / batch_size is truncated.',
                        dest="batch_size")
    # parser.add_argument('--labelled', type=int, action="store", default=False,
    #                     help='If false, you only need one image directory. If true, you need two directories, one with negative samples, the other with positive.',
    #                     dest="preprocess_tfrecs")
    parser.add_argument('--gray_morphology', type=int, action="store", default=True,
                        help='Generate gradcam images with gray morphology. If false, generate gradcam images with red morphology.',
                        dest="gray_morphology")
    parser.add_argument('--imtype', action="store", default='tif',
                        help='suffix for image, tif, jpg, png', dest="imtype")

    args = parser.parse_args()
    print('ARGS:\n', args)

    run_gradcam(args.im_dir, args.resdir, args.deploy_tfrec, args.model_path, args.batch_size, args.layer_name,
                args.gray_morphology, args.imtype)
