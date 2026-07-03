"""
Deploy trained model
Compare original gedi model to newly trained gedi model to human curation
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU execution to avoid CuDNN version conflict

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
import math
try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


__author__ = 'Josh Lamstein'
__copyright__ = 'Gladstone 2021'


class Deploy:
    def __init__(self, parent_dir, preprocess_tfrecs, default_lbl):
        self.default_lbl = default_lbl
        self.parent_dir = parent_dir  
        self.preprocess_tfrecs = preprocess_tfrecs
        if not os.path.exists(self.parent_dir):
            raise FileNotFoundError(f"Parent directory {self.parent_dir} does not exist.")      

    # def run(self, p, im_dir, model_path=None, use_gedi_cnn=True, which_model=None):
    #     deploypath = os.path.join(self.parent_dir, 'deploy.tfrecord')
    #     if self.preprocess_tfrecs:
    #         self.generate_tfrecs(im_dir) 
    #     else:
    #         assert os.path.exists(deploypath), 'set preprocess_tfrecs to true'

    #     self.deploy_main(p, deploypath, model_path, deploy_gedi_cnn=use_gedi_cnn, which_model=which_model)
        
    def run(self, p, im_dir, model_path=None, use_gedi_cnn=True, which_model=None, existing_val_tfrecord=None):
        """
        Run deployment with two modes:
        - preprocess_tfrecs=0: Use existing_val_tfrecord, ignore image_dir
        - preprocess_tfrecs=1: Use image_dir, create tfrecord in parent_dir with label in name
        """

        if self.preprocess_tfrecs:
            # Mode 1: preprocess_tfrecs=1 - Create new tfrecord from image_dir
            print(f"Mode: Creating new TFRecord from images in {im_dir}")
            deploypath = os.path.join(self.parent_dir, 'deploy_{}.tfrecord'.format(self.default_lbl))
            self.generate_tfrecs(im_dir, self.default_lbl)
            print(f"Created TFRecord: {deploypath}")
        else:
            # Mode 2: preprocess_tfrecs=0 - Use existing tfrecord, ignore image_dir
            print(f"Mode: Using existing TFRecord (ignoring image_dir)")
            if existing_val_tfrecord:
                deploypath = existing_val_tfrecord
                print(f"Using provided TFRecord: {deploypath}")
            else:
                # Fall back to default location if no existing tfrecord provided
                deploypath = os.path.join(self.parent_dir, 'deploy_{}.tfrecord'.format(self.default_lbl))
                print(f"Looking for TFRecord at default location: {deploypath}")
                assert os.path.exists(deploypath), f'Validation TFRecord not found at {deploypath}'

        self.deploy_main(p, deploypath, model_path, deploy_gedi_cnn=use_gedi_cnn, which_model=which_model, im_dir=im_dir)

    def generate_tfrecs(self, im_dir, default_lbl):
        """        
        Generate tfrecords from images in im_dir
        Args:
            im_dir: directory with images for analysis
            default_lbl: label to assign to all images

        """
            # Use the passed default_lbl or fall back to self.default_lbl
        lbl = default_lbl

        tfrec_dir = self.parent_dir
        if not os.path.exists(tfrec_dir):
            os.makedirs(tfrec_dir, exist_ok=True)   
        #Rec = Record(im_dir, tfrec_dir, lbl=self.default_lbl)
        Rec = Record(im_dir, tfrec_dir, lbl=lbl) 
        #savedeploy = os.path.join(self.parent_dir, 'deploy.tfrecord')
        ### add label to the saved tfrecord
        savedeploy = os.path.join(self.parent_dir, 'deploy_{}.tfrecord'.format(lbl))
        Rec.tiff2record(savedeploy, Rec.impaths, Rec.lbls)
        print(f'Saved tfrecords to {tfrec_dir}')

    def run_multiclass(self, p, class_dirs, model_path, use_gedi_cnn=False, which_model=None):
        """
        Run deployment over multiple classes in a single pass.
        Args:
            class_dirs: list of (label, im_dir) tuples, e.g. [(0, '/path/c0'), (1, '/path/c1'), (2, '/path/c2')]
        """
        import json

        # Set output_size to match number of classes so one_hot encoding is correct
        p.output_size = len(class_dirs)

        # Step 1: Generate TFRecords for each class
        tfrec_paths = []
        for lbl, im_dir in class_dirs:
            tfrec_path = os.path.join(self.parent_dir, f'deploy_{lbl}.tfrecord')
            if self.preprocess_tfrecs:
                self.generate_tfrecs(im_dir, lbl)
            else:
                assert os.path.exists(tfrec_path), f'TFRecord not found: {tfrec_path}. Set --preprocess_tfrecs 1.'
            tfrec_paths.append((lbl, tfrec_path))

        # Step 2: Load model once
        if use_gedi_cnn:
            p.histogram_eq = False
            p.which_model = 'vgg16'
        else:
            p.histogram_eq = True
            p.which_model = which_model if which_model else 'resnet50'

        print(f'Loading model: {model_path} (architecture: {p.which_model})')
        model = tf.keras.models.load_model(model_path, compile=False)
        for lyr in model.layers:
            lyr.trainable = False
        model.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Step 3: Predict per class, collect results
        all_res = {'filepath': [], 'prediction': [], 'label': []}

        for lbl, tfrec_path in tfrec_paths:
            print(f'\n--- Class {lbl}: {tfrec_path} ---')
            Chk = pipe.Dataspring(p, tfrec_path, False)
            n = int(Chk.count_data().numpy())
            del Chk

            DatTest = pipe.Dataspring(p, tfrec_path, True)
            DatTest.datagen_base(istraining=False)
            test_gen = DatTest.generator()
            DatView = pipe.Dataspring(p, tfrec_path)
            DatView.datagen_base(istraining=False)

            steps_pred = max(1, math.ceil(n / p.BATCH_SIZE))
            res = model.predict(test_gen, steps=steps_pred)

            if res.shape[-1] > 1:
                predictions = np.argmax(res, axis=1)
            else:
                predictions = (res > 0.5).astype(int).squeeze()

            steps_view = steps_pred  # use ceil so last partial batch is not dropped
            for i in tqdm(range(steps_view), desc=f'Class {lbl}'):
                imgs, lbls, _files = DatView.datagen()
                files = _files.numpy()
                nplbls = lbls.numpy()
                batch_preds = predictions[i * p.BATCH_SIZE: min((i + 1) * p.BATCH_SIZE, len(predictions))]
                labels = nplbls if nplbls.ndim == 1 else np.argmax(nplbls, axis=1)
                for t, ell, _file in zip(batch_preds, labels, files):
                    all_res['filepath'].append(_file.decode('utf-8'))
                    all_res['prediction'].append(int(t))
                    all_res['label'].append(int(ell))

        # Step 4: Compute metrics
        res_df = pd.DataFrame(all_res)
        res_df['correct'] = res_df['prediction'] == res_df['label']
        overall_accuracy = res_df['correct'].mean()

        labels_arr = np.array(all_res['label'])
        preds_arr = np.array(all_res['prediction'])
        unique_classes = sorted(set(labels_arr))

        print('\n' + '='*60)
        print('DEPLOY SUMMARY')
        print('='*60)
        print(f'Model    : {model_path}')
        print(f'Arch     : {p.which_model}')
        print('Classes  :')
        for lbl, im_dir in class_dirs:
            print(f'  {lbl} -> {im_dir}')
        print(f'Samples  : {len(res_df)} total')
        print(f'Accuracy : {overall_accuracy * 100:.2f}%')
        print('='*60)

        if SKLEARN_AVAILABLE:
            cm = confusion_matrix(labels_arr, preds_arr, labels=unique_classes)
            report_str = classification_report(labels_arr, preds_arr, labels=unique_classes)
            report_dict = classification_report(labels_arr, preds_arr, labels=unique_classes, output_dict=True)
            print('\nConfusion matrix (rows=true, cols=predicted):')
            print(f"Classes: {unique_classes}")
            print(cm)
            print('\nClassification report:')
            print(report_str)
            cm_list = cm.tolist()
        else:
            # Fallback: manual confusion matrix
            n_cls = len(unique_classes)
            cls_idx = {c: i for i, c in enumerate(unique_classes)}
            cm_list = [[0] * n_cls for _ in range(n_cls)]
            for t, p_ in zip(labels_arr, preds_arr):
                cm_list[cls_idx[t]][cls_idx[p_]] += 1
            report_dict = {}
            print('\nConfusion matrix (rows=true, cols=predicted):')
            print(f"Classes: {unique_classes}")
            for row in cm_list:
                print(row)

        # Step 5: Save outputs
        deploy_results_dir = os.path.join(self.parent_dir, 'deploy_results')
        os.makedirs(deploy_results_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

        csv_path = os.path.join(deploy_results_dir, 'deploy_multiclass.csv')
        if os.path.exists(csv_path):
            csv_path = os.path.join(deploy_results_dir, f'deploy_multiclass_{timestamp}.csv')
        res_df.to_csv(csv_path, index=False)

        summary = {
            'overall_accuracy': float(overall_accuracy),
            'total_samples': int(len(res_df)),
            'correct_predictions': int(res_df['correct'].sum()),
            'confusion_matrix': cm_list,
            'confusion_matrix_classes': unique_classes,
            'classification_report': report_dict,
            'model_path': str(model_path),
            'architecture': str(p.which_model),
            'timestamp': timestamp,
            'class_dirs': {str(lbl): d for lbl, d in class_dirs},
        }
        json_path = os.path.join(deploy_results_dir, 'accuracy_summary_multiclass.json')
        if os.path.exists(json_path):
            json_path = os.path.join(deploy_results_dir, f'accuracy_summary_multiclass_{timestamp}.json')
        def _json_safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_safe)

        print(f'\nFiles saved:')
        print(f'1. Results CSV: {csv_path}')
        print(f'2. Accuracy summary: {json_path}')

    def deploy_main(self, p, deploy_tfrec, model_path, deploy_gedi_cnn, which_model=None, im_dir=None):
        # Don't create a new Param instance, use the one passed in
        res_dict = {'filepath': [], 'prediction': [], 'label': []}

        # if testing on CURATION
        if deploy_gedi_cnn:
            p.histogram_eq = False
            p.which_model = 'vgg16'
            import_path = model_path
        else:
            p.histogram_eq = True  # or whatever you used during training
            p.which_model = which_model if which_model else 'resnet50'  # Use passed parameter
            import_path = model_path
        #print(f'Running model: {import_path}')
        print(f'Running model: {import_path} (architecture: {p.which_model})')
        # p.res_csv_deploy = os.path.join(p.res_dir, 'deploy_results')
        # p.res_csv_deploy = p.res_csv_deploy if p.res_csv_deploy else os.path.join(p.res_dir, 'deploy_results')
        
        print(f"\nOutput directories:")
        print(f"Parent directory: {self.parent_dir}")
        print(f"Results directory: {self.parent_dir}/deploy_results")
    
        if not os.path.exists(p.res_csv_deploy):
            print(f"Creating results directory: {p.res_csv_deploy}")
            os.makedirs(p.res_csv_deploy, exist_ok=True)
        else:
            print(f"Using existing results directory: {p.res_csv_deploy}")

        # Construct paths in deploy_results subfolder
        deploy_results_dir = os.path.join(self.parent_dir, 'deploy_results')
        if not os.path.exists(deploy_results_dir):
            os.makedirs(deploy_results_dir, exist_ok=True)
            
        save_res = os.path.join(deploy_results_dir, 'deploy.csv')
        
        # Plops = plotops.Plotty(model_id)

        # Count samples in tfrecord
        Chk = pipe.Dataspring(p, deploy_tfrec, False)
        test_length = int(Chk.count_data().numpy())
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
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(p.tb_log_dir, p.which_model))

        callbacks = [tb_callback]

        # Predict
        import math
        steps = max(1, math.ceil(test_length / p.BATCH_SIZE))
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

        # for i in range(1):
        # Instead of storing all predictions, process batch by batch, show progress with tqdm
        print('Processing batches...')
        # for i in range(int(test_length // p.BATCH_SIZE)):
        steps = max(1, math.ceil(test_length / p.BATCH_SIZE))  # ceil to include last partial batch
        for i in tqdm(range(steps), desc="Processing batches"):
            #batch_preds = model.predict(next(test_gen))  # Single batch

            # image_batch, lbl_batch = DatTest.datagen()
            # # Plops.show_batch(image_batch, lbl_batch)

            imgs, lbls, _files = DatView.datagen()
            files = _files.numpy()
            nplbls = lbls.numpy()
            test_results = predictions[i * p.BATCH_SIZE: min((i + 1) * p.BATCH_SIZE, len(predictions))]
            # If labels are already single integers, np.argmax
            # labels = np.argmax(nplbls, axis=1)
            labels = nplbls if nplbls.ndim == 1 else np.argmax(nplbls, axis=1)

            for j, (t, ell, _file) in enumerate(zip(test_results, labels, files)):
                file = _file.decode('utf-8')
                res_dict['filepath'].append(file)
                res_dict['prediction'].append(t)
                res_dict['label'].append(ell)
            test_acc = np.array(test_results) == np.array(labels)
            test_acc_batch_avg = np.mean(test_acc)
            test_accuracy_lst.append(test_acc)

            print(f'Batch {i+1}: Accuracy = {test_acc_batch_avg:.3f}')

        # Calculate overall metrics
        #test_accuracy = np.mean(test_accuracy_lst)
        total_samples = len(res_dict['prediction'])
        correct_predictions = sum(np.array(res_dict['prediction']) == np.array(res_dict['label']))
        overall_accuracy = correct_predictions / total_samples
        
        test_accuracy = overall_accuracy


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

        # Add image_dir and label info to metrics
        summary_metrics.update({
            'image_dir': im_dir,
            'default_label': self.default_lbl,
            'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        })

        # Save summary to separate file in deploy_results with timestamp to avoid overwriting
        base_json_name = 'accuracy_summary'
        base_json_path = os.path.join(deploy_results_dir, f'{base_json_name}.json')
        
        # If file exists, create a new one with timestamp
        if os.path.exists(base_json_path):
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            summary_path = os.path.join(deploy_results_dir, f'{base_json_name}_{timestamp}.json')
            print(f"Note: {base_json_name}.json already exists, saving as {os.path.basename(summary_path)}")
        else:
            summary_path = base_json_path
            
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2)

        print(f'\nFiles saved:')
        print(f'1. Accuracy summary (JSON): {summary_path}')

        # Also add accuracy column to the main CSV
        res_df = pd.DataFrame(res_dict)
        res_df['correct'] = res_df['prediction'] == res_df['label']
        
        # Check if CSV exists and create a new one with timestamp if it does
        base_csv_name = 'deploy'
        base_csv_path = os.path.join(deploy_results_dir, f'{base_csv_name}.csv')
        
        if os.path.exists(base_csv_path):
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_res = os.path.join(deploy_results_dir, f'{base_csv_name}_{timestamp}.csv')
            print(f"Note: {base_csv_name}.csv already exists, saving as {os.path.basename(save_res)}")
        else:
            save_res = base_csv_path
            
        res_df.to_csv(save_res, index=False)
        # save csv of results
        print('2. Results data (CSV): {}'.format(save_res))
        #print(f'Percentage of samples that equal {self.default_lbl}:', test_accuracy)
        print(f'Percentage of samples that equal {self.default_lbl} in deploy tfrecord: {test_accuracy * 100:.2f}%')
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
    parser.add_argument('--existing_val_tfrecord', action="store", default=None,
                        help='path to existing validation tfrecord file',
                        dest="existing_val_tfrecord")
    parser.add_argument('--class_dirs', nargs='+', default=None,
                        help='Multiple classes as label:dir pairs, e.g. --class_dirs 0:/path/c0 1:/path/c1 2:/path/c2',
                        dest="class_dirs")

    args = parser.parse_args()
    print('ARGS:\n', args)

    # Use the resdir argument for results, or fall back to parent_dir/deploy_results
    results_dir = args.resdir if args.resdir else os.path.join(args.parent, 'deploy_results')

    p = param.Param(
        parent_dir=args.parent,
        res_dir=results_dir
    )

    if args.class_dirs:
        # Multi-class mode: parse "label:dir" pairs
        class_dirs = []
        for item in args.class_dirs:
            lbl_str, im_dir = item.split(':', 1)
            class_dirs.append((int(lbl_str), im_dir))
        Dep = Deploy(args.parent, args.preprocess_tfrecs, default_lbl=class_dirs[0][0])
        Dep.run_multiclass(p, class_dirs, args.model_path, args.use_gedi_cnn, args.which_model)
    else:
        # Single-class mode (original behaviour)
        Dep = Deploy(args.parent, args.preprocess_tfrecs, args.default_lbl)
        Dep.run(p, args.im_dir, args.model_path, args.use_gedi_cnn, args.which_model, args.existing_val_tfrecord)