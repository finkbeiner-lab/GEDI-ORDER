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

p = param.Param()
SAVE_MONTAGE = False
tfrecord = p.data_deploy
ID_RESULTS = False
CURATION = True
# Load model by setting model_id
model_id = 'vgg16_2020_04_21_10_08_00'  # new data
tp = []
tn = []
fp = []
fn = []
res_dict = {'predictions': []}

# if testing on CURATION
import_path = os.path.join(p.models_dir, "{}.h5".format(model_id))
curation_folder = '/mnt/finkbeinerlab/robodata/GalaxyTEMP/BSMachineLearning_TestCuration/batches/curation_results/v_oza/'
# Get results from original cnn in csv format
orig_cnn_folder = '/mnt/finkbeinerlab/robodata/GalaxyTEMP/BSMachineLearning_TestCuration/batches/curation_results/'
# df = pd.read_csv(os.path.join(curation_folder, 'Batch1_CurationData_29.3761.csv'))
# df = pd.read_csv(os.path.join(curation_folder, 'Batch1_CurationData_VO_15.234.csv'))
# df = pd.read_csv(os.path.join(curation_folder, 'Batch2_CurationData_VO_9.2609.csv'))
# df = pd.read_csv(os.path.join(curation_folder, 'Batch3_CurationData_VO_11.5307.csv'))
df = pd.read_csv(os.path.join(curation_folder, 'Batch4_CurationData_VO_140.0766.csv'))
# df = pd.read_csv(os.path.join(curation_folder, 'Batch5_CurationData_VO_10.4167.csv'))
orig = pd.read_csv(os.path.join(orig_cnn_folder, 'batch4_gedicnn.csv'))

save_res = os.path.join(p.res_csv_deploy, tfrecord.split('/')[-1].split('.')[0] + '.csv')  # save results
Plops = plotops.Plotty(model_id)

# Count samples in tfrecord
Chk = pipe.Dataspring(tfrecord)
test_length = Chk.count_data().numpy()
del Chk
DatTest = pipe.Dataspring(tfrecord)
test_ds = DatTest.datagen_base(istraining=False)
test_gen = DatTest.generator()

DatView = pipe.Dataspring(tfrecord)
view_ds = DatView.datagen_base(istraining=False)

# Load model
print('Loading model...')
model = tf.keras.models.load_model(import_path, compile=False)
for lyr in model.layers:
    lyr.trainable = False
model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
# Predict
res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE)
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
    for j, (t, ell) in enumerate(zip(test_results, labels)):
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
    if ID_RESULTS:
        res_dict['predictions'] = list(predictions)
#
# if SAVE_MONTAGE:
#     if not os.path.exists(p.confusion_dir):
#         os.mkdir(p.confusion_dir)
#     Plops.make_montage(tp, title='True Positives', size=p.BATCH_SIZE)
#     Plops.make_montage(tn, title='True Negatives', size=p.BATCH_SIZE)
#     Plops.make_montage(fp, title='False Positives', size=p.BATCH_SIZE)
#     Plops.make_montage(fn, title='False Negatives', size=p.BATCH_SIZE)
#
# if ID_RESULTS:
#     if not os.path.exists(p.res_csv_deploy):
#         os.mkdir(p.res_csv_deploy)
#     res_df = pd.DataFrame(res_dict)
#     res_df.to_csv(save_res)
if not CURATION:
    print('tp', np.shape(tp))
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
