import os
import platform

from imageio import imread, imwrite
import numpy as np
from subprocess import check_output
import pandas as pd

from transfer.grads import Grads
from transfer.grad_ops import GradOps
import glob
import param_gedi as param
# from memory_profiler import profile
import sys
from pympler import asizeof
import argparse
import pyfiglet
#
# os_type = platform.system()
# if os_type == 'Linux':
#     prefix = '/mnt/finkbeinerlab'
# if os_type == 'Darwin':
#     prefix = '/Volumes/data'


def mem(obj, name):
    m = asizeof.asizeof(obj)
    print(name, m)


def batches_from_fold(source_fold, dead_fold, live_fold, batch_size, parser=lambda x: x):
    # check_files = check_output(['find {}'.format(os.path.join(source_fold, '*.tif'))], shell=True).decode().split()
    check_files = glob.glob(os.path.join(source_fold, '*.tif'))

    files = []
    lbls = []
    for file in check_files:
        fn = file.split('/')[-1]
        cur_lbl = [0] * 2
        if os.path.exists(os.path.join(dead_fold, fn)):
            cur_lbl[0] += 1
        if os.path.exists(os.path.join(live_fold, fn)):
            cur_lbl[1] += 1

        if cur_lbl[0] != cur_lbl[1]:
            files.append(file)
            lbls.append(cur_lbl)
        else:
            print('Couldn\'t find label for image at {}'.format(file))

    for i in range(0, len(files), batch_size):
        print('batch from fold 1')
        _files = files[i:i + batch_size]
        _names = list(map(lambda file: '.'.join(file.split('/')[-1].split('.')[:-1]), _files))
        _lbls = lbls[i:i + batch_size]
        _imgs = list(map(lambda file: parser(imread(file)), _files))
        print('batch from fold 2')

        yield tuple(map(np.array, (_imgs, _lbls, _names)))


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

    ggcam_gen = g.gen_ggcam_stacks(imgs, lbls, layer_name, ret_preds=True)
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
    return res_dict


# @profile
def process_fold(g, source_fold, dead_fold, live_fold, dest_path, conf_mat_paths, batch_size=10, parser=lambda x: x,
                 layer_name='block5_conv3', has_labels=True):
    pred_df = pd.DataFrame({'filename': [], 'label': [], 'prediction': []})
    if has_labels:
        batch_gen = batches_from_fold(source_fold, dead_fold, live_fold, batch_size=batch_size, parser=parser)
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

    print('to csv')

    pred_df.to_csv(os.path.join(dest_path, dest_path.split('/')[-1] + '.csv'))
    return 0

def run_gradcam(main_fold, dest_dir, model_path, layer_name='block5_conv3'):
    # import_path = os.path.join(p.base_gedi_dropout_bn)
    guidedbool = True
    g = Grads(model_path, guidedbool=guidedbool)
    gops = GradOps(vgg_normalize=True)
    conf_mat_paths = [['dead_true', 'dead_false'], ['live_false', 'live_true']]
    dead_fold, live_fold = None, None
    batch_size = 16
    parser = lambda img: gops.img_parse(img)
    # layer_name = 'block5_conv3'
    LABELLED = False
    # layer_name = 'block1_conv1'

    subdirs = glob.glob(os.path.join(main_fold, '**'))
    assert os.path.isdir(subdirs[0]), 'not a subdirectory, check path'
    for subdir in subdirs:
        print('Running {}'.format(subdir))
        name = subdir.split('/')[-1]
        # cur_source_fold = os.path.join(source_fold_prefix, well)
        cur_dest_path = os.path.join(dest_dir, name)

        process_fold(g, subdir, dead_fold, live_fold, cur_dest_path, conf_mat_paths, batch_size=batch_size,
                     parser=parser, layer_name=layer_name, has_labels=LABELLED)


if __name__ == '__main__':
    result = pyfiglet.figlet_format("GEDI-CNN GRADCAM", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Deploy GEDICNN model')
    parser.add_argument('--parent', action="store",
                        default='/run/media/jlamstein/data/GEDI-ORDER',
                        dest='parent')
    parser.add_argument('--im_dir', action="store",
                        default='/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/tmp_deadcrops',
                        help='directory of images to run', dest="im_dir")
    parser.add_argument('--model_path', action="store",
                        # default='/mnt/finkbeinernas/robodata/GEDI_CLUSTER/base_gedi_dropout2.h5',
                        default='/run/media/jlamstein/data/GEDI-ORDER/saved_models/vgg19_2021_11_22_15_00_43.h5',
                        help='path to h5 or hdf5 model', dest="model_path")
    parser.add_argument('--resdir', action="store", default='/mnt/finkbeinernas/robodata/GEDI_CLUSTER/Gradcam',
                        help='results directory', dest="resdir")
    parser.add_argument('--layer_name', action="store", default='block5_conv3',
                        help='visualized layer', dest="layer_name")

    args = parser.parse_args()
    print('ARGS:\n', args)
    # p = param.Param(parent_dir=args.parent, res_dir=args.resdir)
    #
    # import_path = p.base_gedi_dropout
    run_gradcam(args.im_dir,args.resdir, args.model_path, args.layer_name)
