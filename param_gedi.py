"""
Parameter file

Get test data from
/mnt/finkbeinerlab/robodata/GalaxyTEMP/BSMachineLearning_TestCuration/batches
GEDI CNN RESULTS
/mnt/finkbeinerlab/robodata/JeremyTEMP/GEDICNNpaper/GEDImaster/ForJosh.csv
"""

import platform
import os
import datetime
import numpy as np


class Param:
    def __init__(self, param_dict=None, parent_dir=None,  res_dir=None):
        if param_dict is None:
            self.which_model = 'vgg19'  # vgg16, vgg19, resnet50
            self.EPOCHS = 1
            self.learning_rate = 1e-5  # 3e-4
            self.BATCH_SIZE = 32
            self.optimizer = 'adam'  # sgd, adam, adamw
            self.momentum = 0.9
            # Data generator
            self.augmentbool = True
            self.random_brightness = 0.2
            self.min_contrast = 1
            self.max_contrast = 1.3
            self.target_size = (224, 224, 3)
            self.orig_size = (300, 300, 1)  # (230, 230, 3) for catdog tfrecord / (300,300,1) for cells
            self.class_weights = {0: 1., 1: 1.}  # rough ratio  # 2.75 vs 1 for original training dataset
            self.randomcrop = True
            self.histogram_eq = True
            self.weight_decay = 1e-5  # for AdamW
            self.l2_regularize = 0
            self.regularize = None

        else:
            self.which_model = param_dict['model']  # vgg16, vgg19, resnet50
            self.EPOCHS = param_dict['epochs']
            self.learning_rate = param_dict['learning_rate']  # 3e-4
            self.BATCH_SIZE = param_dict['batch_size']
            self.optimizer = param_dict['optimizer']  # sgd, adam
            self.momentum = param_dict['momentum']
            # Data generator
            self.augmentbool = param_dict['augmentbool']
            self.random_brightness = param_dict['random_brightness']
            self.min_contrast = param_dict['min_contrast']
            self.max_contrast = param_dict['max_contrast']
            self.target_size = param_dict['target_size']
            self.orig_size = param_dict['orig_size']  # (230, 230, 3) for catdog tfrecord / (300,300,1) for cells
            self.class_weights = param_dict['class_weights']  # rough ratio  # 2.75 vs 1 for original training dataset
            self.randomcrop = param_dict['randomcrop']
            self.histogram_eq = param_dict['histogram_eq']
            self.weight_decay = param_dict['weight_decay']  # for AdamW
            self.l2_regularize = param_dict['l2_regularize']
            self.regularize = param_dict['regularize']

        self.training_max_value = 1.0001861
        self.training_min_value = 0

        now = datetime.datetime.now()
        self.timestamp = '%d%02d%02d-%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        os_name = platform.node()
        if parent_dir is None:
            self.parent_dir = {'hobbes': '/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER',
                               'calvin': '/mnt/finkbeinerlab/robodata/Josh/GEDI-ORDER',
                               'fb-gpu-compute01.gladstone.internal': '/finkbeiner/imaging/smb-robodata/Josh/GEDI-ORDER',
                               'fb-gpu-compute02.gladstone.internal': '/finkbeiner/imaging/smb-robodata/Josh/GEDI-ORDER'}[
                os_name]
        else:
            self.parent_dir = parent_dir
        # if tfrec_dir is None:
        #     self.tfrec_dir = {
        #         'hobbes': '/mnt/finkbeinernas/robodata/GEDI_CLUSTER/GEDI_DATA',
        #         'calvin': '/run/media/jlamstein/data/gedi/transfer/tfrecs',
        #         'fb-gpu-compute01.gladstone.internal': '/finkbeiner/imaging/smb-robodata/GEDI_CLUSTER/GEDI_DATA/',
        #         'fb-gpu-compute02.gladstone.internal': '/finkbeiner/imaging/smb-robodata/GEDI_CLUSTER/GEDI_DATA/'
        #     }[os_name]
        # else:
        #     self.tfrec_dir = tfrec_dir

        if res_dir is None:
            self.res_dir = {'hobbes': '/mnt/finkbeinernas/robodata/GEDI_CLUSTER',
                            'calvin': '/mnt/finkbeinerlab/robodata/GEDI_CLUSTER',
                            'fb-gpu-compute01.gladstone.internal': '/finkbeiner/imaging/smb-robodata/GEDI_CLUSTER',
                            'fb-gpu-compute02.gladstone.internal': '/finkbeiner/imaging/smb-robodata/GEDI_CLUSTER'}[
                os_name]
        else:
            self.res_dir = res_dir
            self.tfrec_dir = res_dir

        self.base_gedi = os.path.join(self.res_dir, 'gedicnn.h5') # todo: get rid of this?
        self.base_gedi_dropout = os.path.join(self.res_dir, 'base_gedi_dropout2.h5')
        self.base_gedi_dropout_bn = os.path.join(self.res_dir, 'base_gedi_dropout_bn.h5')

        self.run_info_dir = os.path.join(self.parent_dir, 'model_info')
        self.confusion_dir = os.path.join(self.parent_dir, 'confusion_images')
        self.res_csv_deploy = os.path.join(self.parent_dir, 'deploy_results')
        self.models_dir = os.path.join(self.parent_dir, 'saved_models')
        self.ckpt_dir = os.path.join(self.parent_dir, 'saved_checkpoints')
        self.tb_log_dir = os.path.join(self.parent_dir, 'logs')
        self.tfrecord_dir = os.path.join(self.parent_dir, 'TFRECORDS')
        # self.tfrecord_dir = '/mnt/data/CatsAndDogs'
        self.retrain_run_info_dir = os.path.join(self.parent_dir, 'RETRAIN', 'model_info')
        self.retrain_confusion_dir = os.path.join(self.parent_dir, 'RETRAIN', 'confusion_images')
        # self.retrain_res_csv_deploy = os.path.join(self.parent_dir, 'RETRAIN', 'deploy_results')
        self.retrain_models_dir = os.path.join(self.parent_dir, 'RETRAIN', 'SAVED_models')
        self.retrain_ckpt_dir = os.path.join(self.parent_dir, 'RETRAIN', 'saved_checkpoints')

        # self.data_deploy =os.path.join(self.tfrec_dir, 'all_data_val.tfrecords')

        self.orig_train_rec = os.path.join(self.tfrec_dir,
                                           'all_data_train.tfrecords')  # 187772 images {'dead': 50073, 'live': 137699}  2.749
        self.orig_val_rec = os.path.join(self.tfrec_dir,
                                         'all_data_val.tfrecords')  # 20863 images {'dead': 5625, 'live': 15238}  2.708
        self.orig_test_rec = os.path.join(self.tfrec_dir,
                                          'all_data_test.tfrecords')  # 16758 images {'dead': 1372, 'live':15386} 11.2
        self.lincs_train = os.path.join(self.tfrecord_dir, 'LINCS072017RGEDI-A_train.tfrecord')
        self.lincs_val = os.path.join(self.tfrecord_dir, 'LINCS072017RGEDI-A_val.tfrecord')
        self.lincs_test = os.path.join(self.tfrecord_dir, 'LINCS072017RGEDI-A_test.tfrecord')

        self.jk_train = os.path.join(self.tfrec_dir, 'jk_train.tfrecord')
        self.jk_val = os.path.join(self.tfrec_dir, 'jk_val.tfrecord')
        self.jk_test = os.path.join(self.tfrec_dir, 'jk_test.tfrecord')

        self.jk_train_eq = os.path.join(self.tfrec_dir, 'jk_train_eq.tfrecord')
        self.jk_val_eq = os.path.join(self.tfrec_dir, 'jk_val_eq.tfrecord')
        self.jk_test_eq = os.path.join(self.tfrec_dir, 'jk_test_eq.tfrecord')

        self.catdog_train = os.path.join('/run/media/jlamstein/CatsAndDogs/catdog_train.tfrecord')
        self.catdog_val = os.path.join('/run/media/jlamstein/CatsAndDogs/catdog_val.tfrecord')
        self.catdog_test = os.path.join('/run/media/jlamstein/CatsAndDogs/catdog_test.tfrecord')
        #################SET TRAIN AND RETRAIN TFRECORDS#####################
        self.data_retrain = os.path.join(self.tfrecord_dir, 'vor_LINCS092016A_train.tfrecord')
        self.data_reval = os.path.join(self.tfrecord_dir, 'vor_LINCS092016A_val.tfrecord')
        self.data_retest = os.path.join(self.tfrecord_dir, 'vor_LINCS092016A_test.tfrecord')

        # self.data_retrain = os.path.join(self.tfrecord_dir, 'vor_GEDIbiosensor_train.tfrecord')
        # self.data_reval = os.path.join(self.tfrecord_dir, 'vor_GEDIbiosensor_val.tfrecord')
        # self.data_retest = os.path.join(self.tfrecord_dir, 'vor_GEDIbiosensor_test.tfrecord')
        # #
        # self.data_retrain = self.orig_train_rec
        # self.data_reval = self.orig_val_rec
        # self.data_retest = self.orig_test_rec

        self.data_train = self.data_retrain
        self.data_val = self.data_retrain
        self.data_test = self.data_retrain

        # self.data_train = self.lincs_train
        # self.data_val = self.lincs_val
        # self.data_test = self.lincs_test

        # self.data_deploy=self.data_val
        self.save_csv_deploy = ''
        # self.data_deploy = os.path.join(self.tfrecord_dir, 'BSMachineLearning_TestCuration_3.tfrecord')
        self.data_deploy = os.path.join('/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER/testH23.tfrecord')
        # self.data_deploy = self.data_retrain

        # self.max_gedi = 16117. # max value of training set
        self.output_size = 2

        self.orig_width = 300
        self.orig_height = 300
        self.orig_channels = 1
        self.vgg_height = 224
        self.vgg_width = 224
        # self.rescale = 1. / 255

        # self.shuffle_size = 200
        self.num_parallel_calls = 4
        self.num_parallel_reads = 4

        # self.vgg16_weight_path = '/home/jlamstein/Documents/pretrained_weights/vgg16.npy'
        # self.wd_layers = None
        # self.hold_lr = 1e-8
        # self.new_lr = 3e-4
        self.shuffle_buffer_size = 200

        self.hyperparams = {
            'model': self.which_model,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'augmentation': self.augmentbool,
            'random_brightness': self.random_brightness,
            'min_contrast': self.min_contrast,
            'max_contrast': self.max_contrast,
            'train_dir': self.data_train,
            'val_dir': self.data_val,
            'test_dir': self.data_test,
            'orig_size': self.orig_size,
            'target_size': self.target_size,
            'class_weights_0': self.class_weights[0],
            'class_weights_1': self.class_weights[1],
            'batch_size': self.BATCH_SIZE,
            'shuffle_size': self.shuffle_buffer_size,
            'epochs': self.EPOCHS,
            'randomcrop': self.randomcrop,
            'histogram_eq': self.histogram_eq,
            'l2_regularize': self.l2_regularize,
            'regularize': self.regularize,
            'weight_decay': self.weight_decay
        }

        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.AUTOTUNE = True
