"""
Deploy trained model
Compare original gedi model to newly trained gedi model to human curation
"""

import tensorflow as tf
import param_gedi as param
import preprocessing.datagenerator as pipe
import os
import vis.plot_ops as plotops
import numpy as np
import pandas as pd
import argparse


def deploy_main(p, model_id, SAVE_MONTAGE, SAVECSV, CURATION):
    # p = param.Param()
    # SAVE_MONTAGE = False
    tfrecord = p.data_deploy
    SAVECSV = True
    CURATION = False
    # Load model by setting model_id
    # model_id = 'vgg16_2020_04_21_10_08_00'  # new data
    # model_id = 'vgg16_2020_06_25_10_57_54'  # retrained on base_dropuout_bn 06/25/2020
    tp = []
    tn = []
    fp = []
    fn = []
    res_dict = {'filepath': [], 'prediction': [], 'label': []}

    # if testing on CURATION
    # import_path = os.path.join(p.models_dir, "{}.h5".format(model_id))
    import_path = os.path.join(p.ckpt_dir, "{}.hdf5".format(model_id))

    model_name = model_id.split('_')[0]
    if model_name == 'vgg19':
        input_name = 'vgg19_input'
    if model_name == 'resnet50':
        input_name = 'resnet50_input'
    if model_name == 'vgg16':
        input_name = 'input_1'

    # import_path = os.path.join(p.retrain_models_dir, "{}.h5".format(model_id))
    # import_path = os.path.join(p.ckpt_dir, "{}.hdf5".format(model_id))
    # import_path = p.base_gedi_dropout
    # import_path = p.base_gedi_dropout_bn

    if CURATION:
        curation_folder = '/mnt/finkbeinerlab/robodata/GalaxyTEMP/BSMachineLearning_TestCuration/batches/curation_results/v_oza/'
        # Get results from original cnn in csv format
        orig_cnn_folder = '/mnt/finkbeinerlab/robodata/GalaxyTEMP/BSMachineLearning_TestCuration/batches/curation_results/'
        # df = pd.read_csv(os.path.join(curation_folder, 'Batch1_CurationData_29.3761.csv'))
        # df = pd.read_csv(os.path.join(curation_folder, 'Batch1_CurationData_VO_15.234.csv'))
        # df = pd.read_csv(os.path.join(curation_folder, 'Batch2_CurationData_VO_9.2609.csv'))
        # df = pd.read_csv(os.path.join(curation_folder, 'Batch3_CurationData_VO_11.5307.csv'))
        # df = pd.read_csv(os.path.join(curation_folder, 'Batch4_CurationData_VO_140.0766.csv'))
        df = pd.read_csv(os.path.join(curation_folder, 'Batch5_CurationData_VO_10.4167.csv'))
        orig = pd.read_csv(os.path.join(orig_cnn_folder, 'batch5_gedicnn.csv'))

    save_res = os.path.join(p.res_csv_deploy, tfrecord.split('/')[-1].split('.')[0] + '.csv')  # save results
    # Plops = plotops.Plotty(model_id)

    # Count samples in tfrecord
    Chk = pipe.Dataspring(tfrecord, False)
    test_length = Chk.count_data().numpy()
    del Chk
    DatTest = pipe.Dataspring(tfrecord, True)
    test_ds = DatTest.datagen_base(istraining=False)
    test_gen = DatTest.generator(input_name)

    DatView = pipe.Dataspring(tfrecord)
    view_ds = DatView.datagen_base(istraining=False)

    # Load model
    print('Loading model...')
    if 0:
        base_model = tf.keras.models.load_model(import_path, compile=False)
        drop1 = tf.keras.layers.Dropout(rate=0.5, seed=0, name='dropout_1')
        drop2 = tf.keras.layers.Dropout(rate=0.5, seed=0, name='dropout_2')

        fc1 = base_model.get_layer('fc1')
        fc2 = base_model.get_layer('fc2')
        pred_layer = base_model.get_layer('predictions')
        x = drop1(fc1.output)
        x = fc2(x)
        x = drop2(x)
        x = pred_layer(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
        model.save(p.base_gedi_dropout_bn)
    elif 0:
        # todo: remove bn layers and run
        base_model = tf.keras.models.load_model(import_path, compile=False)
        block5_pool = base_model.get_layer('block5_pool')

        drop1 = base_model.get_layer('dropout_1')
        drop2 = base_model.get_layer('dropout_2')
        fc1 = base_model.get_layer('fc1')
        fc2 = base_model.get_layer('fc2')
        fc3 = base_model.get_layer('fc3')
        # output = base_model.get_layer('output')

        x = drop1(fc1.output)
        x = drop1(x)
        x = fc2(x)
        x = drop2(x)
        x = fc3(x)
        # x = output(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    else:
        model = tf.keras.models.load_model(import_path, compile=False)

    for lyr in model.layers:
        lyr.trainable = False
    model.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(p.tb_log_dir, p.which_model),
        update_freq='epoch')

    callbacks = [tb_callback]

    # Predict
    res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE, callbacks=callbacks)
    predictions = np.argmax(res, axis=1)
    test_accuracy_lst = []
    verdict = {'prediction': [], 'curation': [], 'orig_cnn': [], 'Filename': []}
    # for i in range(1):
    for i in range(int(test_length // p.BATCH_SIZE)):
        # image_batch, lbl_batch = DatTest.datagen()
        # # Plops.show_batch(image_batch, lbl_batch)

        imgs, lbls, _files = DatView.datagen()
        files = _files.numpy()
        if CURATION:
            for fdx, _f in enumerate(files):
                f = _f.decode('utf-8')
                f = f.replace('\\', '/')
                f = f.split('/')[-1]

                index = i * p.BATCH_SIZE + fdx
                pred = predictions[index]
                datapoint = df[df.Fname.str.contains(f)]
                orig_datapoint = orig[orig.Fname.str.contains(f)]
                if len(datapoint) > 1:
                    print('multiple rows')
                    datapoint = datapoint.iloc[0]
                verdict['prediction'].append(pred)
                try:
                    verdict['curation'].append(datapoint.Column.values[0])
                    verdict['orig_cnn'].append(orig_datapoint.Column.values[0])
                except AttributeError:
                    verdict['curation'].append(datapoint.Column)
                    verdict['orig_cnn'].append(orig_datapoint.Column)

                verdict['Filename'].append(f)
        nplbls = lbls.numpy()
        if p.output_size == 2:
            # For tfrecs with solid ground truth
            test_results = predictions[i * p.BATCH_SIZE: (i + 1) * p.BATCH_SIZE]
            labels = np.argmax(nplbls, axis=1)
        #         test_results = np.argmax(res[i * p.BATCH_SIZE: (i + 1) * p.BATCH_SIZE], axis=1)
        #         labels = np.argmax(nplbls, axis=1)
        elif p.output_size == 1:
            test_results = np.where(res > 0, 1, 0)
            labels = nplbls
            assert 0
        for j, (t, ell, _file) in enumerate(zip(test_results, labels, files)):
            file = _file.decode('utf-8')
            res_dict['filepath'].append(file)
            res_dict['prediction'].append(t)
            res_dict['label'].append(ell)
            if t == ell:
                if t == 1:
                    tp.append([files[j], lbls[j]])
                else:
                    tn.append([files[j], lbls[j]])
            else:
                if t == 1:
                    fp.append([files[j], lbls[j]])
                else:
                    fn.append([files[j], lbls[j]])
        test_acc = np.array(test_results) == np.array(labels)
        test_acc_batch_avg = np.mean(test_acc)
        test_accuracy_lst.append(test_acc)

    #
    # if SAVE_MONTAGE:
    #     if not os.path.exists(p.confusion_dir):
    #         os.mkdir(p.confusion_dir)
    #     Plops.make_montage(tp, title='True Positives', size=p.BATCH_SIZE)
    #     Plops.make_montage(tn, title='True Negatives', size=p.BATCH_SIZE)
    #     Plops.make_montage(fp, title='False Positives', size=p.BATCH_SIZE)
    #     Plops.make_montage(fn, title='False Negatives', size=p.BATCH_SIZE)
    #
    if SAVECSV:
        if not os.path.exists(p.res_csv_deploy):
            os.mkdir(p.res_csv_deploy)
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(save_res)
        print('Result csv saved to {}'.format(save_res))
    if not CURATION:
        print('tp', len(tp))
        print('tn', len(tn))
        print('fp', len(fp))
        print('fn', len(fn))
        test_accuracy = np.mean(test_accuracy_lst)
        print('test accuracy', test_accuracy)
    else:
        verdict_df = pd.DataFrame(verdict)
        verdict_df.curation = np.abs(verdict_df.curation - 2)
        pred_cur = len(verdict_df[verdict_df.curation == verdict_df.prediction])
        pred_cur_acc = pred_cur / len(verdict_df)
        pred_cnn = len(verdict_df[verdict_df.prediction == verdict_df.orig_cnn])
        pred_cnn_acc = pred_cnn / len(verdict_df)
        cnn_cur = len(verdict_df[verdict_df.orig_cnn == verdict_df.curation])
        cnn_cur_acc = cnn_cur / len(verdict_df)
        print('pos/tot', verdict_df.prediction.sum() / len(verdict_df))
        print('pred vs curation acc', pred_cur_acc)
        print('pred vs orig cnn acc', pred_cnn_acc)
        print('orig cnn vs curation acc', cnn_cur_acc)


if __name__ == '__main__':
    print('Deploying GEDI model...')
    parser = argparse.ArgumentParser(description='Deploy GEDICNN model')
    parser.add_argument('--parent', action="store",
                        default='/mnt/data/GEDI-ORDER',
                        dest='parent')
    parser.add_argument('--tfrecdir', action="store", default='/mnt/finkbeinerlab/robodata/GEDI_CLUSTER/GEDI_DATA',
                        dest="tfrecdir")
    parser.add_argument('--resdir', action="store", default='/mnt/finkbeinerlab/robodata/GEDI_CLUSTER', dest="resdir")
    parser.add_argument('--SAVE_MONTAGE', action="store", default=0, dest="SAVE_MONTAGE")
    parser.add_argument('--SAVECSV', action="store", default='/mnt/finkbeinerlab/robodata/GEDI_CLUSTER', dest="SAVECSV")
    parser.add_argument('--CURATION', action="store", default='/mnt/finkbeinerlab/robodata/GEDI_CLUSTER',
                        dest="CURATION")

    args = parser.parse_args()
    # print('args', args)
    p = param.Param(parent_dir=args.parent, tfrec_dir=args.tfrecdir, res_dir=args.resdir)

    import_path = p.base_gedi_dropout
    # import_path = p.base_gedi_dropout_bn

    deploy_main(p, model_id='vgg19_2021_05_05_15_30_13', SAVE_MONTAGE=args.SAVE_MONTAGE, SAVECSV=args.SAVECSV,
                CURATION=args.CURATION)
