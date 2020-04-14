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

class Param:
    def __init__(self):


        self.which_model = 'vgg16'
        self.EPOCHS = 150
        self.learning_rate= 3e-4
        self.BATCH_SIZE = 16

        now = datetime.datetime.now()
        self.timestamp = '%d%02d%02d-%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

        self.parent_dir = '/mnt/data/GEDI-ORDER'
        os_name = platform.node()
        self.tfrec_dir = {
        'hobbes': '/mnt/data/gedi/transfer/tfrecs',
        'fb-gpu-compute01.gladstone.internal': '/finkbeiner/imaging/smb-robodata/GEDI_CLUSTER/GEDI_DATA/'
        }[os_name]

        self.run_info_dir = os.path.join(self.parent_dir, 'model_info')
        self.confusion_dir = os.path.join(self.parent_dir, 'confusion_images')
        self.res_csv_deploy = os.path.join(self.parent_dir, 'deploy_results')
        self.models_dir = os.path.join(self.parent_dir, 'saved_models')
        self.ckpt_dir = os.path.join(self.parent_dir, 'saved_checkpoints')
        self.tfrecord_dir = os.path.join(self.parent_dir, 'TFRECORDS')

        self.data_deploy = os.path.join(self.tfrecord_dir, 'BSMachineLearning_TestCuration_4.tfrecord')
        # self.data_deploy =os.path.join(self.tfrec_dir, 'all_data_val.tfrecords')

        self.orig_train_rec = os.path.join(self.tfrec_dir,
                                      'all_data_train.tfrecords')  # 187772 images {'dead': 50073, 'live': 137699}  2.749
        self.orig_val_rec = os.path.join(self.tfrec_dir,
                                    'all_data_val.tfrecords')  # 20863 images {'dead': 5625, 'live': 15238}  2.708
        self.orig_test_rec = os.path.join(self.tfrec_dir,
                                     'all_data_test.tfrecords')  # 16758 images {'dead': 1372, 'live':15386} 11.2

        self.jk_train = os.path.join(self.tfrec_dir, 'jk_train.tfrecord')
        self.jk_val = os.path.join(self.tfrec_dir, 'jk_val.tfrecord')
        self.jk_test = os.path.join(self.tfrec_dir, 'jk_test.tfrecord')

        self.jk_train_eq = os.path.join(self.tfrec_dir, 'jk_train_eq.tfrecord')
        self.jk_val_eq = os.path.join(self.tfrec_dir, 'jk_val_eq.tfrecord')
        self.jk_test_eq = os.path.join(self.tfrec_dir, 'jk_test_eq.tfrecord')

        self.data_train = self.orig_train_rec
        self.data_val = self.orig_val_rec
        self.data_test = self.orig_test_rec



        self.class_weights = {0: 2.75, 1: 1.}  # rough ratio  # 2.75 vs 1

        # self.max_gedi = 16117. # max value of training set
        self.output_size = 2
        self.target_size = (224, 224, 3)
        self.orig_size = (300, 300, 1)
        self.orig_width = 300
        self.orig_height = 300
        self.orig_channels = 1
        self.vgg_height = 224
        self.vgg_width = 224
        # self.rescale = 1. / 255
        self.randomcrop = True


        self.shuffle_size = 100
        self.num_parallel_calls = 4
        self.num_parallel_reads = 4

        # self.vgg16_weight_path = '/home/jlamstein/Documents/pretrained_weights/vgg16.npy'
        # self.wd_layers = None
        # self.hold_lr = 1e-8
        # self.new_lr = 3e-4
        self.shuffle_buffer_size = 1000


        #Data generator
        self.augmentbool = True
        self.random_brightness = 0.2
        self.min_contrast = 0.5
        self.max_contrast = 2.0

        self.hyperparams = {
            'timestamp': self.timestamp,
            'train_dir': self.data_train,
            'val_dir': self.data_val,
            'test_dir': self.data_test,
            'class_weights_0': self.class_weights[0],
            'class_weights_1': self.class_weights[1],
            'batch_size': self.BATCH_SIZE,
            'shuffle_size': self.shuffle_size,
            'epochs': self.EPOCHS,
        }


        self.train_transform_params = {
            'rotation_range': 90
            , 'width_shift_range': 0.2
            , 'height_shift_range': 0.2
            , 'shear_range': 0.2
            , 'zoom_range': 0.2
            , 'horizontal_flip': True
            , 'vertical_flip': True
            , 'fill_mode': 'nearest'
        }

        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.AUTOTUNE = True


