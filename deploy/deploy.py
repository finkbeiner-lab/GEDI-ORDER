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
from tqdm import tqdm


__author__ = 'Josh Lamstein'
__copyright__ = 'Gladstone 2021'


class Deploy:
    def __init__(self, parent_dir, preprocess_tfrecs, default_lbl=0):
        self.default_lbl = default_lbl
        self.parent_dir = parent_dir  
        self.preprocess_tfrecs = preprocess_tfrecs
        if not os.path.exists(self.parent_dir):
            raise FileNotFoundError(f"Parent directory {self.parent_dir} does not exist.")      

    def run(self, p, im_dir, model_path=None, use_gedi_cnn=True, which_model=None):
        deploypath = os.path.join(self.parent_dir, 'deploy.tfrecord')
        if self.preprocess_tfrecs:
            self.generate_tfrecs(im_dir) 
        else:
            assert os.path.exists(deploypath), 'set preprocess_tfrecs to true'

        self.deploy_main(p, deploypath, model_path, deploy_gedi_cnn=use_gedi_cnn, which_model=which_model)

    def generate_tfrecs(self, im_dir, default_lbl=None):
        """        
        Generate tfrecords from images in im_dir
        Args:
            im_dir: directory with images for analysis
        """
        tfrec_dir = self.parent_dir
        if not os.path.exists(tfrec_dir):
            os.makedirs(tfrec_dir, exist_ok=True)   
        Rec = Record(im_dir, tfrec_dir, lbl=self.default_lbl)
        savedeploy = os.path.join(self.parent_dir, 'deploy.tfrecord')
        Rec.tiff2record(savedeploy, Rec.impaths, Rec.lbls)
        print(f'Saved tfrecords to {tfrec_dir}')

    def deploy_main(self, p, deploy_tfrec, model_path, deploy_gedi_cnn, which_model=None):
        # p = param.Param()
        res_dict = {'filepath': [], 'prediction': [], 'label': []}

        # if testing on CURATION
        if deploy_gedi_cnn:
            p.histogram_eq = False
            p.which_model = 'vgg16'
            import_path = model_path
        else:
            p.histogram_eq = False  # or whatever you used during training
            p.which_model = which_model if which_model else 'resnet50'  # Use passed parameter
            import_path = model_path
        #print(f'Running model: {import_path}')
        print(f'Running model: {import_path} (architecture: {p.which_model})')
        #p.res_csv_deploy = os.path.join(p.res_dir, 'deploy_results')
        p.res_csv_deploy = os.path.join(p.res_dir, 'predictions')
        #p.res_csv_deploy = p.res_csv_deploy if p.res_csv_deploy else os.path.join(p.res_dir, 'deploy_results')
        if not os.path.exists(p.res_csv_deploy):
            os.makedirs(p.res_csv_deploy, exist_ok=True)        

        save_res = os.path.join(p.res_csv_deploy, deploy_tfrec.split('/')[-1].split('.')[0] + '.csv')
        
        # Plops = plotops.Plotty(model_id)

        # Count samples in tfrecord
        Chk = pipe.Dataspring(p, deploy_tfrec, False)
        test_length = Chk.count_data().numpy()
        del Chk
        DatTest = pipe.Dataspring(p, deploy_tfrec, True)
        test_ds = DatTest.datagen_base(istraining=False)
        test_gen = DatTest.generator()

        DatView = pipe.Dataspring(p, deploy_tfrec)
        view_ds = DatView.datagen_base(istraining=False)

        # Load model
        print('Loading model...')
        # if deploy_gedi_cnn:
        #     # remove batchnorm layers and run
        #     base_model = tf.keras.models.load_model(import_path, compile=False)
        #     block5_pool = base_model.get_layer('block5_pool')
        #
        #     drop1 = base_model.get_layer('dropout_1')
        #     drop2 = base_model.get_layer('dropout_2')
        #     fc1 = base_model.get_layer('fc1')
        #     fc2 = base_model.get_layer('fc2')
        #     fc3 = base_model.get_layer('fc3')
        #
        #     x = drop1(fc1.output)
        #     x = drop1(x)
        #     x = fc2(x)
        #     x = drop2(x)
        #     x = fc3(x)
        #     model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
        # else:
        #     model = tf.keras.models.load_model(import_path, compile=False)
        model = tf.keras.models.load_model(import_path, compile=False)

        for lyr in model.layers:
            lyr.trainable = False
        model.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(p.tb_log_dir, p.which_model))

        callbacks = [tb_callback]

        # Predict
        import math
        steps = max(1, math.ceil(test_length / p.BATCH_SIZE))
        #steps = max(1, int(test_length // p.BATCH_SIZE))
        print(f"Test length: {test_length}, Batch size: {p.BATCH_SIZE}, Steps: {steps}")
        res = model.predict(test_gen, steps=steps)
        #res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE)
        # for binary classification with sigmoid, predictions would be shape (N, 1) and should be:
        # predictions = np.argmax(res, axis=1)
        if res.shape[-1] > 1:
            predictions = np.argmax(res, axis=1)  # Multi-class
        else:
            predictions = (res > 0.5).astype(int).squeeze()  # Binary
        test_accuracy_lst = []
        verdict = {'prediction': [], 'curation': [], 'orig_cnn': [], 'Filename': []}

        # Instead of storing all predictions, process batch by batch, show progress with tqdm
        # for i in range(int(test_length // p.BATCH_SIZE)):

        print('Processing batches...')
        steps = max(1, math.ceil(test_length / p.BATCH_SIZE))
        #steps = max(1, int(test_length // p.BATCH_SIZE))

        print(f"Predictions array length: {len(predictions)}")
        print(f"Expected samples: {steps * p.BATCH_SIZE}")

        for i in tqdm(range(steps), desc="Processing batches"):    
            # Get batch data from the view generator
            imgs, lbls, _files = DatView.datagen()
            files = _files.numpy()
            nplbls = lbls.numpy()
            # Get the corresponding predictions for this batch
            start_idx = i * p.BATCH_SIZE
            end_idx = min((i + 1) * p.BATCH_SIZE, len(predictions))
            test_results = predictions[start_idx:end_idx]
            # Process labels
            labels = nplbls if nplbls.ndim == 1 else np.argmax(nplbls, axis=1)
                   
            # for j, (t, ell, _file) in enumerate(zip(test_results, labels, files)):
            #     file = _file.decode('utf-8')
            #     res_dict['filepath'].append(file)
            #     res_dict['prediction'].append(t)
            #     res_dict['label'].append(ell)
            # test_acc = np.array(test_results) == np.array(labels)
            # test_acc_batch_avg = np.mean(test_acc)
            # test_accuracy_lst.append(test_acc)
            # Store results for each sample in the batch
            for j, (t, ell, _file) in enumerate(zip(test_results, labels, files)):
                if j < len(test_results):  # Safety check
                    file = _file.decode('utf-8')
                    res_dict['filepath'].append(file)
                    res_dict['prediction'].append(int(t))  # Ensure it's a Python int
                    res_dict['label'].append(int(ell))     # Ensure it's a Python int
            
            # Calculate batch accuracy
            test_acc = np.array(test_results) == np.array(labels[:len(test_results)])
            test_acc_batch_avg = np.mean(test_acc)
            test_accuracy_lst.append(test_acc)

            print(f'Batch {i+1}: Accuracy = {test_acc_batch_avg:.3f}')

        # Calculate overall metrics
        #test_accuracy = np.mean(test_accuracy_lst)
        total_samples = len(res_dict['prediction'])
        correct_predictions = sum(np.array(res_dict['prediction']) == np.array(res_dict['label']))
        overall_accuracy = correct_predictions / total_samples
        
        # Now calculate batch-level accuracy properly
        test_accuracy = overall_accuracy  # Use the actual overall accuracy

        # Create summary metrics
        summary_metrics = {
            'total_samples': int(total_samples),
            'correct_predictions': int(correct_predictions), 
            'overall_accuracy': float(overall_accuracy),
            'batch_accuracies': [float(np.mean(batch_acc)) for batch_acc in test_accuracy_lst],
            'mean_batch_accuracy': float(np.mean([np.mean(batch_acc) for batch_acc in test_accuracy_lst])),
            'model_path': str(import_path),
            'architecture': str(p.which_model)
        }

        # Save summary to separate file
        summary_path = os.path.join(p.res_csv_deploy, 'accuracy_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2)

        print(f'Accuracy summary saved to: {summary_path}')

        # Also add accuracy column to the main CSV
        res_df = pd.DataFrame(res_dict)
        res_df['correct'] = res_df['prediction'] == res_df['label']
        res_df.to_csv(save_res, index=False)
        # save csv of results
        print('Result csv saved to {}'.format(save_res))
        #print(f'Percentage of samples that equal {self.default_lbl}:', test_accuracy)
        print(f'Percentage of samples that equal {self.default_lbl} in deploy tfrecord: {test_accuracy * 100:.2f}%')
        print(f'Mean batch accuracy on deploy tfrecord: {test_accuracy * 100:.2f}%')


if __name__ == '__main__':
    result = pyfiglet.figlet_format("DEPLOY CNN", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Deploy GEDICNN model')
    parser.add_argument('--parent', action="store",
                        default='/Users/swang/FinkbeinerLab/GEDI-CNN',
                        help='parent directory for Gedi-CNN',
                        dest='parent')
    parser.add_argument('--im_dir', action="store",
                        default='/Volumes/Finkbeiner-Linsley/Shijie/Galaxy-temp-CRY2tau/GXYTMP-10222021-AAVtau-216-EVall/ObjectCrops',
                        help='directory of images to run', dest="im_dir")
    parser.add_argument('--model_path', action="store",
                        default='/Users/swang/FinkbeinerLab/GEDI-CNN/gedicnn.h5',
                        help='path to h5 or hdf5 model', dest="model_path")
    parser.add_argument('--resdir', action="store", default='/Users/swang/FinkbeinerLab/GEDI-CNN/DeployResults',
                        help='results directory', dest="resdir")
    parser.add_argument('--preprocess_tfrecs', type=int, action="store", default=True,
                        help='generate tfrecords, necessary for new datasets, if already generate set to false',
                        dest="preprocess_tfrecs")
    parser.add_argument('--use_gedi_cnn', type=int, action="store", default=True,
                        help='generate tfrecords, necessary for new datasets, if already generate set to false',
                        dest="use_gedi_cnn")
    parser.add_argument('--which_model', type=str, action="store", default="resnet50",
                        help='which model to use, ResNet50 or VGG19 or VGG16',
                        dest="which_model")
    parser.add_argument('--default_lbl', type=int, action="store", default=0,
                        help='default label for tfrecords, 0 for negative, 1 for positive',
                        dest="default_lbl")
                        

    args = parser.parse_args()
    # args.preprocess_tfrecs and args.default_lbl are already correct types (int)
    print('ARGS:\n', args)
    p = param.Param(parent_dir=args.parent, res_dir=args.resdir)

    Dep = Deploy(args.parent, args.preprocess_tfrecs, args.default_lbl)
    Dep.run(p, args.im_dir, args.model_path, args.use_gedi_cnn, args.which_model)