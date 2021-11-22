"""
Train model


"""

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
from preprocessing.create_tfrecs import Record
import pyfiglet

__author__ = 'Josh Lamstein'
__copyright__ = 'Gladstone Institutes 2020'


class Train:
    def __init__(self, parent_dir, preprocess_tfrecs):
        self.parent_dir = parent_dir
        self.preprocess_tfrecs = preprocess_tfrecs

    def run(self, pos_dir, neg_dir, balance_method='cutoff'):
        if self.preprocess_tfrecs:
            self.generate_tfrecs(pos_dir, neg_dir, balance_method)
        else:
            assert os.path.exists(os.path.join(self.parent_dir, 'train.tfrecord')), 'set preprocess_tfrecs to true'
        self.train()

    def run_retrain(self, pos_dir, neg_dir, balance_method='cutoff'):
        if self.preprocess_tfrecs:
            self.generate_tfrecs(pos_dir, neg_dir, balance_method)
        else:
            assert os.path.exists(os.path.join(self.parent_dir, 'retrain.tfrecord')), 'set preprocess_tfrecs to true'
        self.retrain()

    def generate_tfrecs(self, pos_dir, neg_dir, balance_method='cutoff'):
        """
        Builds tfrecords from images sorted in directories by label
        Args:
            pos_dir: directory with positive images
            neg_dir: directory with negative images
            balance_method: method to handle unbalanced datasets, default set to cutoff larger samples

        Returns:

        """
        split = [.7, .15, .15]
        tfrec_dir = self.parent_dir
        Rec = Record(pos_dir, neg_dir, tfrec_dir, split, balance_method=balance_method)
        savetrain = 'train.tfrecord'
        saveval = 'val.tfrecord'
        savetest = 'test.tfrecord'
        Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
        Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
        Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
        print(f'Saved tfrecords to {tfrec_dir}')

    def train(self):

        print('Running...')
        # Setup filepaths and csv to log info about training model
        p = param.Param(parent_dir=self.parent_dir)
        make_directories(p)
        tfrec_dir = self.parent_dir
        data_train = os.path.join(tfrec_dir, 'train.tfrecord')
        data_val = os.path.join(tfrec_dir, 'val.tfrecord')
        data_test = os.path.join(tfrec_dir, 'test.tfrecord')

        timestamp = update_timestring()
        export_path = os.path.join(p.models_dir, '{}_{}.h5'.format(p.which_model, timestamp))
        export_info_path = os.path.join(p.run_info_dir, '{}_{}.csv'.format(p.which_model, timestamp))
        save_checkpoint_path = os.path.join(p.ckpt_dir, '{}_{}.hdf5'.format(p.which_model, timestamp))
        run_info = {'model': p.which_model,
                    'retraining': '',
                    'timestamp': timestamp,
                    'model_timestamp': p.which_model + '_' + timestamp,
                    'train_path': data_train,
                    'val_path': data_val,
                    'test_path': data_test,
                    'learning_rate': p.learning_rate,
                    'augmentation': p.augmentbool,
                    'epochs': p.EPOCHS,
                    'batch_size': p.BATCH_SIZE,
                    'output_size': p.output_size,
                    'im_shape': p.target_size,
                    'random_crop': p.randomcrop}
        # Get length of tfrecords
        Chk = pipe.Dataspring(data_train)
        train_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(data_val)
        val_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(data_test)
        test_length = Chk.count_data().numpy()
        del Chk
        DatTrain = pipe.Dataspring(data_train)
        DatVal = pipe.Dataspring(data_val)
        DatTest = pipe.Dataspring(data_test)
        DatTest2 = pipe.Dataspring(data_test)

        train_ds = DatTrain.datagen_base(istraining=True)
        val_ds = DatVal.datagen_base(istraining=True)
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

        net = CNN()
        if p.which_model == 'vgg16':
            model = net.vgg16(imsize=p.target_size)
        elif p.which_model == 'vgg19':
            model = net.vgg19(imsize=p.target_size)
        elif p.which_model == 'mobilenet':
            model = net.mobilenet(imsize=p.target_size)
        elif p.which_model == 'inceptionv3':
            model = net.inceptionv3(imsize=p.target_size)
        else:
            model = net.standard_model(imsize=p.target_size)

        # callbacks, save checkpoints and tensorboard logs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(save_checkpoint_path, monitor='val_accuracy', verbose=1,
                                                         save_best_only=True, mode='max')

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(p.tb_log_dir, p.which_model),
            update_freq='epoch')

        callbacks = [cp_callback]
        history = model.fit(train_gen, steps_per_epoch=train_length // (p.BATCH_SIZE), epochs=p.EPOCHS,
                            class_weight=p.class_weights, validation_data=val_gen,
                            validation_steps=val_length // p.BATCH_SIZE, callbacks=callbacks)
        # history = model.fit(train_gen, steps_per_epoch=train_length // (p.BATCH_SIZE), epochs=p.EPOCHS,
        #                     validation_data=val_gen,
        #                     validation_steps=val_length // p.BATCH_SIZE, callbacks=callbacks)

        # for _ in range(10):
        #     train_ims, train_lbls = DatTrain.datagen()
        #     val_ims, val_lbls = DatVal.datagen()
        #     history = model.fit(train_ims, train_lbls, steps_per_epoch=p.BATCH_SIZE, epochs=1, validation_data=(val_ims, val_lbls),
        #                         validation_steps=p.BATCH_SIZE, callbacks=callbacks)
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        print('Evaluating model...')

        # Predict on test dataset
        res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE)
        test_accuracy_lst = []
        # Get accuracy, compare predictions with labels
        for i in range(int(test_length // p.BATCH_SIZE)):
            imgs, lbls, files = DatTest2.datagen()
            # res = model.predict((imgs, lbls), steps=test_length // p.BATCH_SIZE, workers=4, use_multiprocessing=True)
            nplbls = lbls.numpy()
            if p.output_size == 2:
                test_results = np.argmax(res[i * p.BATCH_SIZE: (i + 1) * p.BATCH_SIZE], axis=1)
                labels = np.argmax(nplbls, axis=1)
            elif p.output_size == 1:
                test_results = np.where(res > 0, 1, 0)
                labels = nplbls
            test_acc = np.array(test_results) == np.array(labels)
            test_acc_batch_avg = np.mean(test_acc)
            test_accuracy_lst.append(test_acc)

        test_accuracy = np.mean(test_accuracy_lst)
        print('test accuracy', test_accuracy)
        run_info['train_accuracy'] = train_acc[-1]
        run_info['val_accuracy'] = val_acc[-1]
        run_info['test_accuracy'] = test_accuracy
        run_info['train_loss'] = train_loss[-1]
        run_info['val_loss'] = val_loss[-1]

        run_df = pd.DataFrame([run_info])
        run_df.to_csv(export_info_path)

        print('Saving model to {}'.format(export_path))
        model.save(export_path)

    def retrain(self, base_model=None):

        print('Running...')
        # Setup filepaths and csv to log info about training model
        p = param.Param(parent_dir=self.parent_dir)
        make_directories(p)
        if base_model is None:  # load base model to initialize weights
            base_model = p.base_gedi_dropout_bn

        tfrec_dir = self.parent_dir
        data_retrain = os.path.join(tfrec_dir, 'retrain.tfrecord')
        data_reval = os.path.join(tfrec_dir, 'reval.tfrecord')
        data_retest = os.path.join(tfrec_dir, 'retest.tfrecord')
        timestamp = update_timestring()
        export_path = os.path.join(p.retrain_models_dir, '{}_{}.h5'.format(p.which_model, timestamp))
        export_info_path = os.path.join(p.retrain_run_info_dir, '{}_{}.csv'.format(p.which_model, timestamp))
        save_checkpoint_path = os.path.join(p.retrain_ckpt_dir, '{}_{}.hdf5'.format(p.which_model, timestamp))
        run_info = {'model': p.which_model,
                    'retraining': p.base_gedi_dropout_bn,
                    'timestamp': timestamp,
                    'model_timestamp': p.which_model + '_' + timestamp,
                    'train_path': data_retrain,
                    'val_path': data_reval,
                    'test_path': data_retest,
                    'learning_rate': p.learning_rate,
                    'augmentation': p.augmentbool,
                    'epochs': p.EPOCHS,
                    'batch_size': p.BATCH_SIZE,
                    'output_size': p.output_size,
                    'im_shape': p.target_size,
                    'random_crop': p.randomcrop}
        # Get length of tfrecords
        Chk = pipe.Dataspring(data_retrain)
        train_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(data_reval)
        val_length = Chk.count_data().numpy()
        del Chk
        Chk = pipe.Dataspring(data_retest)
        test_length = Chk.count_data().numpy()
        del Chk
        DatTrain = pipe.Dataspring(data_retrain)
        DatVal = pipe.Dataspring(data_reval)
        DatTest = pipe.Dataspring(data_retest)
        DatTest2 = pipe.Dataspring(data_retest)

        train_ds = DatTrain.datagen_base(istraining=True)
        val_ds = DatVal.datagen_base(istraining=True)
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
        base_model = tf.keras.models.load_model(base_model, compile=False)
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

        glorot = tf.initializers.GlorotUniform()
        # bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, name='bn_1')
        # bn2 = tf.keras.layers.BatchNormalization(momentum=0.9, name='bn_2')
        # fc1_small = tf.keras.layers.Dense(64, name='fc1', activation='relu', kernel_initializer='TruncatedNormal',
        #                    bias_initializer='TruncatedNormal')
        # fc2_small = tf.keras.layers.Dense(16, name='fc2', activation='relu', kernel_initializer='TruncatedNormal',
        #                    bias_initializer='TruncatedNormal')
        # prediction = tf.keras.layers.Dense(p.output_size, activation='softmax', name='output')
        #
        # drop1 = base_model.get_layer('dropout_1')
        # drop2 = base_model.get_layer('dropout_2')
        # block5_pool = base_model.get_layer('block5_pool')
        # flatten = tf.keras.layers.Flatten()
        #
        # #
        # fc1 = base_model.get_layer('fc1')
        # fc2 = base_model.get_layer('fc2')
        # pred_layer = base_model.get_layer('predictions')
        #
        # x = flatten(block5_pool.output)
        # x = fc1_small(x)
        # x = drop1(x)
        # x = fc2_small(x)
        # x = drop2(x)
        # x = prediction(x)
        # model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
        model = base_model
        for lyr in model.layers:

            if 'block5' in lyr.name or 'fc1' in lyr.name or 'fc2' in lyr.name or 'dropout' in lyr.name:
                _weights = lyr.get_weights()
                # if len(_weights) > 0:
                #     print('resetting weights:', lyr.name)
                #     W = np.shape(_weights[0])
                #     b = np.shape(_weights[1])
                #     lyr.set_weights([glorot(shape=W), glorot(shape=b)])
                lyr.trainable = True
            else:
                lyr.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        # callbacks, save checkpoints and tensorboard logs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(save_checkpoint_path, monitor='val_accuracy', verbose=1,
                                                         save_best_only=True, mode='max')

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='/home/jlamstein/PycharmProjects/ASYN/log/{}'.format(p.which_model),
            update_freq='epoch')

        callbacks = [cp_callback]
        history = model.fit(train_gen, steps_per_epoch=train_length // (p.BATCH_SIZE), epochs=p.EPOCHS,
                            class_weight=p.class_weights, validation_data=val_gen,
                            validation_steps=val_length // p.BATCH_SIZE, callbacks=callbacks)

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
        res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE)
        test_accuracy_lst = []
        # Get accuracy, compare predictions with labels
        for i in range(int(test_length // p.BATCH_SIZE)):
            imgs, lbls, files = DatTest2.datagen()
            nplbls = lbls.numpy()
            if p.output_size == 2:  # output size is two. One hot encoded binary classifier
                test_results = np.argmax(res[i * p.BATCH_SIZE: (i + 1) * p.BATCH_SIZE], axis=1)
                labels = np.argmax(nplbls, axis=1)
            elif p.output_size == 1:  # only for output size one, so far not used
                test_results = np.where(res > 0, 1, 0)
                labels = nplbls
            test_acc = np.array(test_results) == np.array(labels)
            test_acc_batch_avg = np.mean(test_acc)
            test_accuracy_lst.append(test_acc)

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


if __name__ == '__main__':
    result = pyfiglet.figlet_format("GEDI-CNN", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Train binary classifer GEDI-CNN model')
    parser.add_argument('--datadir', action="store",
                        default='/run/media/jlamstein/data/GEDI-ORDER',
                        help='data parent directory',
                        dest='datadir')
    parser.add_argument('--pos_dir', action="store",
                        default='/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Livecrops_3',
                        help='directory with positive images', dest="pos_dir")
    parser.add_argument('--neg_dir', action="store",
                        default='/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Deadcrops_3',
                        help='directory with negative images', dest="neg_dir")
    parser.add_argument('--balance_method', action="store", default='cutoff',
                        help='method to handle unbalanced data: cutoff, multiply or none', dest="balance_method")
    parser.add_argument('--preprocess_tfrecs', type=bool, action="store", default=False,
                        help='generate tfrecords, necessary for new datasets, if already generate set to false',
                        dest="preprocess_tfrecs")
    args = parser.parse_args()
    print('ARGS:\n', args)

    Tr = Train(args.datadir, args.preprocess_tfrecs)
    Tr.run(args.pos_dir, args.neg_dir, args.balance_method)
