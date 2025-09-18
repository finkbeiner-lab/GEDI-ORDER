"""
Train model
Setting train/test/val from three different paths

"""

import os
# Force TensorFlow to use CPU due to CuDNN version mismatch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import param_gedi as param
from models.model import CNN
import preprocessing.datagenerator as pipe
from utils.utils import update_timestring, make_directories
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from preprocessing.create_tfrecs_from_lst import Record
import pyfiglet
from glob import glob
import random
import tensorflow_addons as tfa
import wandb
from wandb.integration.keras import (
    WandbMetricsLogger,
    WandbModelCheckpoint,
    WandbEvalCallback
)


__author__ = 'Finkbeiner Lab'
__copyright__ = 'Gladstone Institutes 2025'


class Train:
    def __init__(self, parent_dir, param_dict=None, preprocess_tfrecs=False, use_wandb=True):
        self.parent_dir = parent_dir
        self.p = param.Param(param_dict=param_dict, parent_dir=self.parent_dir)
        self.preprocess_tfrecs = preprocess_tfrecs
        self.use_wandb = use_wandb
        if self.use_wandb:
            run = wandb.init(
            #entity="swang202",
            #project="TauKO_mito_ko_vs_oeTau",
            config={
                "learning_rate": self.p.learning_rate,
                "architecture": self.p.which_model,
                "dataset": "TauPFF",
                "epochs": self.p.EPOCHS,
            },
)
            csv = os.path.join(self.p.parent_dir, 'wandb.csv')
            if os.path.exists(csv):
                df = pd.read_csv(csv)
            else:
                df = None
            # self.nep = neptune.init(df['user'].iloc[0], df['token'].iloc[0])
        else:
            self.use_wandb = None

    def run(self, pos_dirs, neg_dirs, balance_method='cutoff'):
        assert isinstance(pos_dirs, list), 'pos_dirs must be list'

        # add split_method, original: 'percentage'
        if self.preprocess_tfrecs or not os.path.exists(os.path.join(self.parent_dir, 'test.tfrecord')):
            self.generate_tfrecs(pos_dirs, neg_dirs, balance_method, split_method='percentage')
        else:
            assert os.path.exists(os.path.join(self.parent_dir, 'train.tfrecord')), 'set preprocess_tfrecs to true'
        self.train()

    def run_retrain(self, pos_dir, neg_dir, balance_method='cutoff'):
        if self.preprocess_tfrecs:
            self.generate_tfrecs(pos_dir, neg_dir, balance_method)
        else:
            assert os.path.exists(os.path.join(self.parent_dir, 'retrain.tfrecord')), 'set preprocess_tfrecs to true'
        self.retrain()

    def gather_imgs(self, pos_dirs, neg_dirs, filetype='tif'):
        """Gather images from matching subfolders in live/dead directories
        
        Directory structure:
        live/
        ├── T0_0/*.tif
        ├── T1_12/*.tif
        └── T12_144/*.tif
        dead/
        ├── T0_0/*.tif
        ├── T1_12/*.tif
        └── T12_144/*.tif
        """
        pos_ims = {}  # Dictionary to store images by timepoint
        neg_ims = {}
        
        # Get all timepoint folders
        timepoints = []
        for pos_dir in pos_dirs:
            timepoints.extend([f.name for f in os.scandir(pos_dir) if f.is_dir()])
        timepoints = sorted(list(set(timepoints)))  # Unique, sorted timepoints
        
        print(f"\nFound {len(timepoints)} timepoints: {timepoints}")
        
        # Gather images for each timepoint
        for tp in timepoints:
            pos_ims[tp] = []
            neg_ims[tp] = []
            
            # Get positive/live images for this timepoint
            for pos_dir in pos_dirs:
                tp_dir = os.path.join(pos_dir, tp)
                if os.path.exists(tp_dir):
                    pos_ims[tp].extend(glob(os.path.join(tp_dir, f'*.{filetype}')))
                    
            # Get negative/dead images for this timepoint
            for neg_dir in neg_dirs:
                tp_dir = os.path.join(neg_dir, tp)
                if os.path.exists(tp_dir):
                    neg_ims[tp].extend(glob(os.path.join(tp_dir, f'*.{filetype}')))
            
            # Shuffle images for this timepoint
            random.Random(11).shuffle(pos_ims[tp])
            random.Random(11).shuffle(neg_ims[tp])
            
            print(f"\nTimepoint {tp}:")
            print(f"  Found {len(pos_ims[tp])} live images")
            print(f"  Found {len(neg_ims[tp])} dead images")
            if pos_ims[tp]:
                print(f"  Sample live: {pos_ims[tp][0]}")
            if neg_ims[tp]:
                print(f"  Sample dead: {neg_ims[tp][0]}")
    
        return pos_ims, neg_ims, timepoints

    def generate_tfrecs(self, pos_dirs, neg_dirs, split, balance_method='cutoff', split_method='percentage'):
        """
        Builds tfrecords from images sorted in directories by label
        Args:
            pos_dir: directory with positive images
            neg_dir: directory with negative images
            balance_method: method to handle unbalanced datasets, default set to cutoff larger samples

        Returns:

        """

        split = [.7, .15, .15]  # Required parameter, ignored when split_method='tiles'
        tfrec_dir = self.parent_dir
        pos_ims_dict, neg_ims_dict, timepoints = self.gather_imgs(pos_dirs, neg_dirs, filetype='tif')

        # Flatten the dictionaries into lists for Record class
        pos_ims = []
        neg_ims = []
        for tp in timepoints:
            pos_ims.extend(pos_ims_dict[tp])
            neg_ims.extend(neg_ims_dict[tp])

        print(f"DEBUG: Sample image paths:")
        print(f"Total positive images: {len(pos_ims)}")
        print(f"Total negative images: {len(neg_ims)}")
        if pos_ims:
            print(f"Sample positive: {pos_ims[:2]}")
        if neg_ims:
            print(f"Sample negative: {neg_ims[:2]}")

        # SW: Too much dadta - set Cutoff to max_images_per_class
        # Shuffle deterministically using a fixed seed
        # random.seed(42)
        # random.shuffle(pos_ims)
        # random.shuffle(neg_ims)
        # pos_ims = sorted(pos_ims)[:10000]
        # neg_ims = sorted(neg_ims)[:10000]

        Rec = Record(pos_ims, neg_ims, tfrec_dir, split, balance_method, split_method=split_method)
        savetrain = 'train.tfrecord'
        saveval = 'val.tfrecord'
        savetest = 'test.tfrecord'
        Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
        Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
        Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
        print(f'Saved tfrecords to {tfrec_dir}')

    def train(self):
        print('Running...')
        tf.keras.backend.clear_session()
        # Setup filepaths and csv to log info about training model
        make_directories(self.p)
        tfrec_dir = self.parent_dir
        data_train = os.path.join(tfrec_dir, 'train.tfrecord')
        data_val = os.path.join(tfrec_dir, 'val.tfrecord')
        data_test = os.path.join(tfrec_dir, 'test.tfrecord')
        assert os.path.exists(data_train), 'check tfrecord path and that tfrecord exists'

        timestamp = update_timestring()
        #export_path = os.path.join(self.p.models_dir, '{}_{}.h5'.format(self.p.which_model, timestamp))
        export_path = os.path.join(self.p.models_dir, '{}_{}.keras'.format(self.p.which_model, timestamp))

        export_info_path = os.path.join(self.p.run_info_dir, '{}_{}.csv'.format(self.p.which_model, timestamp))
        #changed .hdf5 to .keras
        save_checkpoint_path = os.path.join(self.p.ckpt_dir, '{}_{}.keras'.format(self.p.which_model, timestamp))
        self.p.hyperparams['timestamp'] = timestamp
        self.p.hyperparams['model_timestamp'] = self.p.which_model + '_' + timestamp
        self.p.hyperparams['retraining'] = ''
        # if self.use_neptune:
        #     self.nep["parameters"] = self.p.hyperparams

        # todo replace with self.p.hyperparams
        run_info = {'model': self.p.which_model,
                    'retraining': '',
                    'timestamp': timestamp,
                    'model_timestamp': self.p.which_model + '_' + timestamp,
                    'train_path': data_train,
                    'val_path': data_val,
                    'test_path': data_test,
                    'learning_rate': self.p.learning_rate,
                    'augmentation': self.p.augmentbool,
                    'epochs': self.p.EPOCHS,
                    'batch_size': self.p.BATCH_SIZE,
                    'output_size': self.p.output_size,
                    'im_shape': self.p.target_size,
                    'random_crop': self.p.randomcrop}
        # Get length of tfrecords
        Chk = pipe.Dataspring(self.p, data_train)
        train_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(self.p, data_val)
        val_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(self.p, data_test)
        test_length = Chk.count_data().numpy()
        del Chk
        DatTrain = pipe.Dataspring(self.p, data_train)
        DatVal = pipe.Dataspring(self.p, data_val)
        DatTest = pipe.Dataspring(self.p, data_test)
        DatTest2 = pipe.Dataspring(self.p, data_test)

        train_ds = DatTrain.datagen_base(istraining=True)
        val_ds = DatVal.datagen_base(istraining=False)
        test_ds = DatTest.datagen_base(istraining=False)
        test_ds2 = DatTest2.datagen_base(istraining=False)
        print('training length', train_length)
        print('validation length', val_length)
        print('test length', test_length)
        for _key, _val in run_info.items():
            print(f'{_key} : {_val}')
        train_gen = DatTrain.generator()
        val_gen = DatVal.generator()
        test_gen = DatTest.generator()

        # for image_batch, label_batch in val_ds.take(1):
        #     print('min img', np.min(image_batch))
        #     for img, lbl in zip(image_batch, label_batch):
        #         plt.figure()
        #         im = (img + 1) /2
        #         plt.imshow(im)
        #         plt.title(lbl.numpy())
        #
        # plt.show()

        net = CNN(self.p)
        if self.p.which_model == 'vgg16':
            model = net.vgg16(imsize=self.p.target_size)
        elif self.p.which_model == 'vgg19':
            model = net.vgg19(imsize=self.p.target_size)
        elif self.p.which_model == 'resnet50':
            model = net.resnet50(imsize=self.p.target_size)
        elif self.p.which_model == 'custom2':
            model = net.custom_model2(imsize=self.p.target_size)
        elif self.p.which_model == 'custom1':
            model = net.custom_model(imsize=self.p.target_size)
        else:
            assert 0, 'no model specified'

        # callbacks, save checkpoints and tensorboard logs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(save_checkpoint_path, monitor='val_accuracy', verbose=1,
                                                         save_best_only=True, mode='max')

        cp_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.001, patience=3, verbose=1,
            mode='min', baseline=None, restore_best_weights=True
        )

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.p.tb_log_dir, self.p.which_model),
            update_freq='epoch')
        
        #callbacks = [cp_callback, cp_early]
        callbacks = [cp_callback, cp_early, tb_callback]

        if self.use_wandb:
            # Get unique subfolder names by reading from the tfrecord files
            subfolders = set()
            # We'll get subfolders from the file structure instead
            # since pos_ims and neg_ims are not available in this scope
            
            wandb_callbacks = [
                WandbMetricsLogger(log_freq='batch'),
                WandbModelCheckpoint(
                    filepath=save_checkpoint_path,
                    monitor="val_accuracy",
                    save_best_only=True
                )
            ]
            callbacks += wandb_callbacks


        # if self.use_neptune:
        #     from neptune.new.integrations.tensorflow_keras import NeptuneCallback
        #     neptune_cbk = NeptuneCallback(run=self.nep, base_namespace='metrics')
        #     callbacks.append(neptune_cbk)
        print(f"Train length: {train_length}, Val length: {val_length}, Batch size: {self.p.BATCH_SIZE}")
        history = model.fit(train_gen, steps_per_epoch=int(train_length // self.p.BATCH_SIZE), epochs=self.p.EPOCHS,
                             #class_weight=self.p.class_weights, 
                             validation_data=val_gen,
                             validation_steps=int(val_length // self.p.BATCH_SIZE), callbacks=callbacks)

        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        print('Evaluating model...')

        # Predict on test dataset
        res = model.predict(test_gen, steps=int(test_length // self.p.BATCH_SIZE))
        test_accuracy_lst = []
        # Get accuracy, compare predictions with labels
        for i in range(int(test_length // self.p.BATCH_SIZE)):
            imgs, lbls, files = DatTest2.datagen()
            # res = model.predict((imgs, lbls), steps=test_length // self.p.BATCH_SIZE, workers=4, use_multiprocessing=True)
            nplbls = lbls.numpy()
            if self.p.output_size == 2:
                test_results = np.argmax(res[i * self.p.BATCH_SIZE: (i + 1) * self.p.BATCH_SIZE], axis=1)
                labels = np.argmax(nplbls, axis=1)
            elif self.p.output_size == 1:
                test_results = np.where(res > 0, 1, 0)
                labels = nplbls
            test_acc = np.array(test_results) == np.array(labels)
            if isinstance(test_acc, bool):
                test_acc = [test_acc]
            else:
                test_acc = list(test_acc)
            test_accuracy_lst.extend(test_acc)

        test_accuracy = np.mean(test_accuracy_lst)
        print('test accuracy', test_accuracy)
        run_info['train_accuracy'] = train_acc[-1]
        run_info['val_accuracy'] = val_acc[-1]
        run_info['test_accuracy'] = test_accuracy
        run_info['train_loss'] = train_loss[-1]
        run_info['val_loss'] = val_loss[-1]
        # if self.use_neptune:
        #     self.p.hyperparams['test_acc'] = test_accuracy
        #     self.nep['parameters'] = self.p.hyperparams

        run_df = pd.DataFrame([run_info])
        run_df.to_csv(export_info_path)

        print('Saving model to {}'.format(export_path))
        model.save(export_path)


    def retrain(self, base_model_file=None):

        print('Running...')
        tf.keras.backend.clear_session()
        # Setup filepaths and csv to log info about training model
        make_directories(self.p)
        if base_model_file is None:  # load base model to initialize weights
            base_model_file = self.p.base_gedi
            self.p.which_model = 'vgg16'
            self.p.histogram_eq = False

        tfrec_dir = self.parent_dir
        data_retrain = os.path.join(tfrec_dir, 'train.tfrecord')
        data_reval = os.path.join(tfrec_dir, 'val.tfrecord')
        data_retest = os.path.join(tfrec_dir, 'test.tfrecord')
        timestamp = update_timestring()
        export_path = os.path.join(self.p.retrain_models_dir, '{}_{}.h5'.format(self.p.which_model, timestamp))
        export_info_path = os.path.join(self.p.retrain_run_info_dir, '{}_{}.csv'.format(self.p.which_model, timestamp))
        save_checkpoint_path = os.path.join(self.p.retrain_ckpt_dir, '{}_{}.hdf5'.format(self.p.which_model, timestamp))
        self.p.hyperparams['model'] = self.p.which_model
        self.p.hyperparams['timestamp'] = timestamp
        self.p.hyperparams['model_timestamp'] = self.p.which_model + '_' + timestamp
        self.p.hyperparams['retraining'] = base_model_file
        # if self.use_neptune:
        #     self.nep["parameters"] = self.p.hyperparams
        run_info = {'model': self.p.which_model,
                    'retraining': base_model_file,
                    'timestamp': timestamp,
                    'model_timestamp': self.p.which_model + '_' + timestamp,
                    'train_path': data_retrain,
                    'val_path': data_reval,
                    'test_path': data_retest,
                    'learning_rate': self.p.learning_rate,
                    'augmentation': self.p.augmentbool,
                    'epochs': self.p.EPOCHS,
                    'batch_size': self.p.BATCH_SIZE,
                    'output_size': self.p.output_size,
                    'im_shape': self.p.target_size,
                    'random_crop': self.p.randomcrop}
        # Get length of tfrecords
        Chk = pipe.Dataspring(self.p, data_retrain)
        train_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(self.p, data_reval)
        val_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(self.p, data_retest)
        test_length = Chk.count_data().numpy()
        del Chk
        DatTrain = pipe.Dataspring(self.p, data_retrain)
        DatVal = pipe.Dataspring(self.p, data_reval)
        DatTest = pipe.Dataspring(self.p, data_retest)
        DatTest2 = pipe.Dataspring(self.p, data_retest)

        train_ds = DatTrain.datagen_base(istraining=True)
        val_ds = DatVal.datagen_base(istraining=False)
        test_ds = DatTest.datagen_base(istraining=False)
        test_ds2 = DatTest2.datagen_base(istraining=False)
        print('training length', train_length)
        print('validation length', val_length)
        print('test length', test_length)
        for _key, _val in run_info.items():
            print(f'{_key} : {_val}')

        # uses retraining generator

        train_gen = DatTrain.generator()
        val_gen = DatVal.generator()
        test_gen = DatTest.generator()
        test_gen2 = DatTest2.generator()

        print('Loading model...')
        base_model = tf.keras.models.load_model(base_model_file, compile=False)
        # visualize kernels to check model weights
        # if rapid chagne and plateu check data, biases chenged with grad descent
        # check base model included in graph and weigths are  changing
        # Check that live = 1, dead = 0
        # verify that new data live/dead look accurate
        # try different learning rates
        # vgg16 lesion some layers with random weights, initialize with imagenet, train with 10000 images
        # Mismatch between labels and data, dead=dead, live = live
        # no black images
        # contrast range is similar, contrast normalization skimage, histogram equalization.
        # histogram matching of old vs new data histogram normalization
        # represent every image by mean intensity
        # learn affine transformation, scale and intercept, on average transform human to rat

        glorot = tf.initializers.GlorotUniform
        trunc = tf.initializers.TruncatedNormal

        bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, name='bn_1')
        bn2 = tf.keras.layers.BatchNormalization(momentum=0.9, name='bn_2')
        fc1_small = tf.keras.layers.Dense(256, name='fc1', activation='relu', kernel_initializer=trunc(),
                                          bias_initializer=trunc())
        fc2_small = tf.keras.layers.Dense(256, name='fc2', activation='relu', kernel_initializer=trunc(),
                                          bias_initializer=trunc())
        prediction = tf.keras.layers.Dense(self.p.output_size, activation='softmax', name='output')

        # drop1 = base_model.get_layer('dropout_1')
        # drop2 = base_model.get_layer('dropout_2')
        block5_pool = base_model.get_layer('block5_pool')
        flatten = tf.keras.layers.Flatten()

        #
        # fc1 = base_model.get_layer('fc1')
        # fc2 = base_model.get_layer('fc2')
        # pred_layer = base_model.get_layer('predictions')

        x = flatten(block5_pool.output)
        x = fc1_small(x)
        # x = drop1(x)
        x = fc2_small(x)
        # x = drop2(x)
        x = prediction(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

        # model = base_model
        for lyr in model.layers:

            if 'block3' in lyr.name or 'block4' in lyr.name or 'block5' in lyr.name or 'fc1' in lyr.name or 'fc2' in lyr.name or 'dropout' in lyr.name:
                # _weights = lyr.get_weights()
                # if len(_weights) > 0:
                #     print('resetting weights:', lyr.name)
                #     W = np.shape(_weights[0])
                #     b = np.shape(_weights[1])
                #     lyr.set_weights([glorot(shape=W), glorot(shape=b)])
                lyr.trainable = True
            else:
                lyr.trainable = False
        if self.p.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate)
        elif self.p.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.p.learning_rate, momentum=self.p.momentum,
                                                nesterov=True)
        elif self.p.optimizer == 'adamw':
            optimizer = tfa.optimizers.AdamW(learning_rate=self.p.learning_rate, weight_decay=self.p.weight_decay)

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        # callbacks, save checkpoints and tensorboard logs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(save_checkpoint_path, monitor='val_accuracy', verbose=1,
                                                         save_best_only=True, mode='max')

        cp_early = tf.keras.callbacks.EarlyStopping(
            monitor='loss', min_delta=0, patience=3, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True
        )
        callbacks = [cp_callback, cp_early]

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='/home/jlamstein/PycharmProjects/ASYN/log/{}'.format(self.p.which_model),
            update_freq='epoch' # Change from 'epoch' to 'batch'
            )
        
        if self.use_wandb:
            from wandb.integration.keras import (
                WandbMetricsLogger,
                WandbModelCheckpoint,
                WandbEvalCallback
            )

            class MyEvalCallback(WandbEvalCallback):
                def add_ground_truth(self, epoch, logs=None):
                    import numpy as np
                    images, labels = next(iter(val_gen))
                    table = wandb.Table(columns=["image", "label"])
                    for i in range(min(32, len(images))):
                        img = images[i].numpy() if hasattr(images[i], 'numpy') else images[i]
                        lbl = labels[i].numpy() if hasattr(labels[i], 'numpy') else labels[i]
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        # Handle different label formats
                        if lbl.ndim > 0 and lbl.size > 1:
                            # One-hot encoded or multi-dimensional - get the argmax
                            scalar_lbl = int(np.argmax(lbl))
                        else:
                            # Already scalar or single element
                            scalar_lbl = int(lbl.item() if hasattr(lbl, 'item') else lbl)
                        table.add_data(wandb.Image(img), scalar_lbl)
                    wandb.log({"ground_truth": table})

                def add_model_predictions(self, epoch, logs=None):
                    import numpy as np
                    images, labels = next(iter(val_gen))
                    preds = self.model.predict(images)
                    pred_labels = preds.argmax(axis=1) if preds.shape[-1] > 1 else (preds > 0.5).astype(int)
                    table = wandb.Table(columns=["image", "prediction"])
                    for i in range(min(32, len(images))):
                        img = images[i].numpy() if hasattr(images[i], 'numpy') else images[i]
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        scalar_pred = int(pred_labels[i].item() if hasattr(pred_labels[i], 'item') else pred_labels[i])
                        table.add_data(wandb.Image(img), scalar_pred)
                    wandb.log({"predictions": table})


            wandb_callbacks = [
                WandbMetricsLogger(
                    log_freq='batch',
                    log_additional_metrics={
                        'live_accuracy': None,
                        'dead_accuracy': None,
                    }
                ),
                WandbModelCheckpoint(
                    filepath=save_checkpoint_path,
                    monitor="val_accuracy",
                    save_best_only=True
                ),
                MyEvalCallback(
                    data_table_columns=["image", "label"],
                    pred_table_columns=["image", "prediction"]
                )
            ]
            callbacks += wandb_callbacks

        history = model.fit(train_gen, steps_per_epoch=train_length // (self.p.BATCH_SIZE), epochs=self.p.EPOCHS,
                             #class_weight=self.p.class_weights, 
                             validation_data=val_gen,
                             validation_steps=int(val_length // self.p.BATCH_SIZE), callbacks=callbacks)

        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        print('Evaluating model...')
        # Freeze whole model for evaluation
        for lyr in model.layers:
            lyr.trainable = False
        model.trainable = False

        # Predict on test dataset
        res = model.predict(test_gen, steps=int(test_length // self.p.BATCH_SIZE))
        test_accuracy_lst = []
        # Get accuracy, compare predictions with labels
        for i in range(int(test_length // self.p.BATCH_SIZE)):
            imgs, lbls, files = DatTest2.datagen()
            nplbls = lbls.numpy()
            if self.p.output_size == 2:  # output size is two. One hot encoded binary classifier
                test_results = np.argmax(res[i * self.p.BATCH_SIZE: (i + 1) * self.p.BATCH_SIZE], axis=1)
                labels = np.argmax(nplbls, axis=1)
            elif self.p.output_size == 1:  # only for output size one, so far not used
                test_results = np.where(res > 0, 1, 0)
                labels = nplbls
            test_acc = np.array(test_results) == np.array(labels)
            if isinstance(test_acc, bool):
                test_acc = [test_acc]
            else:
                test_acc = list(test_acc)
            test_accuracy_lst.extend(test_acc)

        test_accuracy = np.mean(test_accuracy_lst)
        print('test accuracy', test_accuracy)
        # Add test info to output csv
        run_info['train_accuracy'] = train_acc[-1]
        run_info['val_accuracy'] = val_acc[-1]
        run_info['test_accuracy'] = test_accuracy
        run_info['train_loss'] = train_loss[-1]
        run_info['val_loss'] = val_loss[-1]

        run_df = pd.DataFrame([run_info])
        run_df.to_csv(export_info_path)

        print('Saving model to {}'.format(export_path))
        model.save(export_path)


class SubfolderEvalCallback(WandbEvalCallback):
    def __init__(self, val_gen, subfolder_pairs, **kwargs):
        super().__init__(**kwargs)
        self.val_gen = val_gen
        self.subfolder_pairs = subfolder_pairs  # Dict mapping subfolder names to their file indices
        
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        
        # Get predictions on validation set
        images, labels, filenames = next(iter(self.val_gen))
        preds = self.model.predict(images)
        pred_labels = preds.argmax(axis=1) if preds.shape[-1] > 1 else (preds > 0.5).astype(int)
        true_labels = labels.numpy().argmax(axis=1) if labels.numpy().ndim > 1 else labels.numpy()
        
        # Track metrics per subfolder
        subfolder_metrics = {}
        for subfolder in self.subfolder_pairs:
            # Get indices for this subfolder's files
            subfolder_mask = [f.decode('utf-8').split('/')[-2] == subfolder for f in filenames]
            if not any(subfolder_mask):
                continue
                
            # Calculate accuracies for this subfolder
            subfolder_preds = pred_labels[subfolder_mask]
            subfolder_true = true_labels[subfolder_mask]
            
            live_mask = subfolder_true == 1
            dead_mask = subfolder_true == 0
            
            subfolder_metrics[subfolder] = {
                'accuracy': np.mean(subfolder_preds == subfolder_true),
                'live_accuracy': np.mean(subfolder_preds[live_mask] == subfolder_true[live_mask]) if any(live_mask) else 0,
                'dead_accuracy': np.mean(subfolder_preds[dead_mask] == subfolder_true[dead_mask]) if any(dead_mask) else 0
            }
            
            # Log to wandb
            wandb.log({
                f"{subfolder}/accuracy": subfolder_metrics[subfolder]['accuracy'],
                f"{subfolder}/live_accuracy": subfolder_metrics[subfolder]['live_accuracy'],
                f"{subfolder}/dead_accuracy": subfolder_metrics[subfolder]['dead_accuracy'],
                f"{subfolder}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=subfolder_true,
                    preds=subfolder_preds,
                    class_names=["Dead", "Live"]
                )
            }, commit=False)
        
        # Log overall metrics
        wandb.log({
            "overall/accuracy": np.mean(pred_labels == true_labels),
            "overall/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_labels,
                preds=pred_labels,
                class_names=["Dead", "Live"]
            )
        })


if __name__ == '__main__':
    result = pyfiglet.figlet_format("CNN", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Train binary classifer CNN model')
    positives = ['/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-1703/Livecrops_1', '/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-1703/Livecrops_2_3']
    negatives = ['/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-1703/Deadcrops_1', '/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-1703/Deadcrops_2_3']
    parser.add_argument('--datadir', action="store",
                        default='/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER',
                        help='data parent directory',
                        dest='datadir')
    parser.add_argument('--pos_dir', nargs='+',
                        default=positives,
                        help='directory with positive images', dest="pos_dir")
    parser.add_argument('--neg_dir', nargs='+',
                        default=negatives,
                        help='directory with negative images', dest="neg_dir")
    parser.add_argument('--balance_method', action="store", default='cutoff',
                        help='method to handle unbalanced data: cutoff, multiply or none', dest="balance_method")
    parser.add_argument('--preprocess_tfrecs', type=str, action="store", default=False,
                        help='generate tfrecords, necessary for new datasets, if already generate set to false',
                        dest="preprocess_tfrecs")
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD optimizer (if used)')
    parser.add_argument('--use_wandb', type=int, action="store", default=True,
                        help='Save run info to wandb',
                        dest="use_wandb")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--which_model', default='vgg19', type=str)
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='Optimizer to use: adam, sgd, adamw, etc.', dest='optimizer')
    parser.add_argument('--learning_rate', default=0.0005, type=float,  
                        help='Learning rate')
    parser.add_argument('--retrain', type=int, action="store", default=False,
                        help='Save run info to wandb',
                        dest="retrain")
    args = parser.parse_args()

    # Convert string "0"/"1" to boolean for preprocess_tfrecs
    args.preprocess_tfrecs = bool(int(args.preprocess_tfrecs))
    args.use_wandb = bool(int(args.use_wandb))
    
    print('ARGS:\n', args)

    # SW: make sure args is passed to Train
    param_dict = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,  
        'model': args.which_model,      
        'optimizer': args.optimizer,
        'balance_method': args.balance_method,
        'momentum': 0.9       
    }

    Tr = Train(parent_dir=args.datadir, param_dict=param_dict, preprocess_tfrecs=args.preprocess_tfrecs,
               use_wandb=args.use_wandb)
    if args.retrain:
        print('Retraining on gedi-cnn model...')
        Tr.retrain()
    else:
        Tr.run(args.pos_dir, args.neg_dir, args.balance_method)  # generates tfrecs if arg is set to true and trains
