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
from preprocessing.create_tfrecs_deploy import Record
import pyfiglet


class Deploy:
    def __init__(self):
        self.default_lbl = 0

    def run(self, p, im_dir, model_path=None):
        self.generate_tfrecs(im_dir)
        self.deploy_main(p, model_path, deploy_gedi_cnn=True if model_path is None else False)

    def generate_tfrecs(self, im_dir):
        """

        Args:
            im_dir: directory with images for analysis
        """
        tfrec_dir = os.getcwd()
        Rec = Record(im_dir, tfrec_dir, lbl=self.default_lbl)
        savedeploy = 'deploy.tfrecord'
        Rec.tiff2record(savedeploy, Rec.impaths, Rec.lbls)
        print(f'Saved tfrecords to {tfrec_dir}')

    def deploy_main(self, p, model_path, deploy_gedi_cnn):
        # p = param.Param()
        tfrecord = p.data_deploy
        res_dict = {'filepath': [], 'prediction': [], 'label': []}

        # if testing on CURATION
        if model_path is None:
            import_path = p.base_gedi_dropout
        else:
            import_path = model_path

        save_res = os.path.join(p.res_csv_deploy, tfrecord.split('/')[-1].split('.')[0] + '.csv')  # save results
        # Plops = plotops.Plotty(model_id)

        # Count samples in tfrecord
        Chk = pipe.Dataspring(tfrecord, False)
        test_length = Chk.count_data().numpy()
        del Chk
        DatTest = pipe.Dataspring(tfrecord, True)
        test_ds = DatTest.datagen_base(istraining=False)
        test_gen = DatTest.generator()

        DatView = pipe.Dataspring(tfrecord)
        view_ds = DatView.datagen_base(istraining=False)

        # Load model
        print('Loading model...')
        if deploy_gedi_cnn:
            # remove batchnorm layers and run
            base_model = tf.keras.models.load_model(import_path, compile=False)
            block5_pool = base_model.get_layer('block5_pool')

            drop1 = base_model.get_layer('dropout_1')
            drop2 = base_model.get_layer('dropout_2')
            fc1 = base_model.get_layer('fc1')
            fc2 = base_model.get_layer('fc2')
            fc3 = base_model.get_layer('fc3')

            x = drop1(fc1.output)
            x = drop1(x)
            x = fc2(x)
            x = drop2(x)
            x = fc3(x)
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
            nplbls = lbls.numpy()
            test_results = predictions[i * p.BATCH_SIZE: (i + 1) * p.BATCH_SIZE]
            labels = np.argmax(nplbls, axis=1)
            for j, (t, ell, _file) in enumerate(zip(test_results, labels, files)):
                file = _file.decode('utf-8')
                res_dict['filepath'].append(file)
                res_dict['prediction'].append(t)
                res_dict['label'].append(ell)
            test_acc = np.array(test_results) == np.array(labels)
            test_acc_batch_avg = np.mean(test_acc)
            test_accuracy_lst.append(test_acc)

        # save csv of results
        if not os.path.exists(p.res_csv_deploy):
            os.mkdir(p.res_csv_deploy)
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(save_res)
        print('Result csv saved to {}'.format(save_res))

        test_accuracy = np.mean(test_accuracy_lst)
        print(f'Percentage of samples that equal {self.default_lbl}', test_accuracy)


if __name__ == '__main__':
    result = pyfiglet.figlet_format("GEDI-CNN", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Deploy GEDICNN model')
    parser.add_argument('--parent', action="store",
                        default='/mnt/data/GEDI-ORDER',
                        dest='parent')
    parser.add_argument('--im_dir', action="store", default='/mnt/finkbeinerlab/robodata/GEDI_CLUSTER',
                        help='directory of images to run', dest="im_dir")
    parser.add_argument('--model_path', action="store",
                        default='/mnt/finkbeinerlab/robodata/GEDI_CLUSTER/base_gedi_dropout2.h5',
                        help='path to h5 or hdf5 model', dest="model_path")
    parser.add_argument('--resdir', action="store", default='/mnt/finkbeinerlab/robodata/GEDI_CLUSTER',
                        help='results directory', dest="resdir")

    args = parser.parse_args()
    print('ARGS:\n', args)
    p = param.Param(parent_dir=args.parent, res_dir=args.resdir)

    import_path = p.base_gedi_dropout
    Dep = Deploy()
    Dep.run(p, args.im_dir, args.model_path)
