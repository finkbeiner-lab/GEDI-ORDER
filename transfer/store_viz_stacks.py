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

os_type = platform.system()
if os_type == 'Linux':
    prefix = '/mnt/finkbeinerlab'
if os_type == 'Darwin':
    prefix = '/Volumes/data'


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


p = param.Param()

# g = Grads(prefix + '/robodata/Gennadi/tf_to_k_v2.h5')
# timestamp = 'vgg16_2020_04_20_15_05_47' #2drop, 2bn
# timestamp = 'vgg16_2020_04_21_10_08_00' #1drop, 2bn
# import_path = os.path.join(p.models_dir, "{}.h5".format(timestamp))
import_path = os.path.join(p.base_gedi_dropout_bn)
g = Grads(import_path)
gops = GradOps(vgg_normalize=True)

main_fold = prefix + '/robodata/GalaxyTEMP/BSMachineLearning_TestCuration'
source_fold_prefix = os.path.join(main_fold, 'batches')
# dead_fold = os.path.join(main_fold, 'master', 'DEAD')
# live_fold = os.path.join(main_fold, 'master', 'LIVE')
# dest_path_prefix = prefix + '/robodata/Gennadi/batches_grads2'
# conf_mat_paths = [['dead_true', 'dead_false'], ['live_false', 'live_true']]


# main_fold = prefix + '/robodata/Josh/Gradcam/ObjectCrops'
# main_fold = prefix + '/robodata/JeremyTEMP/GEDICNNpaper/HumanIncorrectCNNcorrectImages'
# '/robodata/JaslinTemp/GalaxyData/LINCS-diMNs/LINCS072017RGEDI-A/Galaxy-wholeplate/Galaxy/CroppedImages/LINCS072017RGEDI-A'
# main_fold = prefix + '/robodata/JeremyTEMP/GalaxyTEMP/ShortTimeGEDI/ObjectCrops'
# source_fold_prefix = main_fold
dead_fold = os.path.join(main_fold, 'master', 'DEAD')
live_fold = os.path.join(main_fold, 'master', 'LIVE')
dest_path_prefix = prefix + '/robodata/Josh/Gradcam/results/batches_grads_2020-5-18'
conf_mat_paths = [['dead_true', 'dead_false'], ['live_false', 'live_true']]

batch_size = 10
parser = lambda img: gops.img_parse(img)
layer_name = 'block5_conv3'
LABELLED=True
# layer_name = 'block1_conv1'

# # Example usage
# process_fold(g, source_fold, dead_fold, live_fold, dest_path, conf_mat_paths, batch_size=batch_size, parser=parser, layer_name=layer_name)
subdirs = glob.glob(os.path.join(main_fold, '**'))
wells = [w.split('/')[-1] for w in subdirs]
# wells = [w for w in wells if w not in ['E5', 'B2', 'H10', 'B8', 'F7', 'H9', 'H3', 'C1', 'B10', 'E11', 'G4', 'F12', 'G12', 'D11', 'G9', 'G3', 'C10']]
# wells = ['HumanIncorrectDeadNoInnerSoma', 'HumanIncorrectLiveNoInnerSoma', 'HumanCorrectLiveInnerSoma']
for well in map(str, range(3, 20 + 1)):
    # for well in wells:
    print('Running {}'.format(well))
    cur_source_fold = os.path.join(source_fold_prefix, well)
    cur_dest_path = os.path.join(dest_path_prefix, well)

    process_fold(g, cur_source_fold, dead_fold, live_fold, cur_dest_path, conf_mat_paths, batch_size=batch_size,
                 parser=parser, layer_name=layer_name, has_labels=LABELLED)
