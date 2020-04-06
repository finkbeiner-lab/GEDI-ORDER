import tensorflow as tf
import param_gedi as param
from models.model import CNN
import preprocessing.datagenerator as pipe
from ops.processing_ops import get_tfrecord_length
import os
import visualization.plot_ops as plotops
import datetime
from utils.utils import update_timestring
import numpy as np
import pandas as pd

p = param.Param()
SAVE_MONTAGE= False
tfrecord = p.data_deploy
ID_RESULTS = True
timestamp = 'vgg16_2020_03_24_17_06_40'  # new data
tp = []
tn = []
fp = []
fn = []
res_dict = {'predictions': []}
import_path = os.path.join(p.models_dir, "{}.h5".format(timestamp))

save_res = os.path.join(p.res_csv_deploy, tfrecord.split('/')[-1].split('.')[0] + '.csv')
Plops = plotops.Plotty(timestamp)

Chk = pipe.Dataspring(tfrecord)
test_length = Chk.count_data().numpy()
del Chk
DatTest = pipe.Dataspring(tfrecord)
test_ds = DatTest.datagen_base(istraining=False)
test_gen = DatTest.generator()

print('Loading model...')
model = tf.keras.models.load_model(import_path)
res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE)
predictions = np.argmax(res, axis=1)
test_accuracy_lst = []
for i in range(int(test_length // p.BATCH_SIZE)):
    # image_batch, lbl_batch = DatTest.datagen()
    # # Plops.show_batch(image_batch, lbl_batch)

    imgs, lbls, files = DatTest.datagen()
    nplbls = lbls.numpy()
    if p.output_size == 2:
        test_results = np.argmax(res[i * p.BATCH_SIZE: (i + 1) * p.BATCH_SIZE], axis=1)
        labels = np.argmax(nplbls, axis=1)
    elif p.output_size == 1:
        test_results = np.where(res > 0, 1, 0)
        labels = nplbls
    for j, (t, ell) in enumerate(zip(test_results, labels)):
        if t == ell:
            if t == 1:
                tp.append([imgs[j], lbls[j]])
            else:
                tn.append([imgs[j], lbls[j]])
        else:
            if t == 1:
                fp.append([imgs[j], lbls[j]])
            else:
                fn.append([imgs[j], lbls[j]])
    test_acc = np.array(test_results) == np.array(labels)
    test_acc_batch_avg = np.mean(test_acc)
    test_accuracy_lst.append(test_acc)
    if ID_RESULTS:
        res_dict['predictions'] = list(predictions)


if SAVE_MONTAGE:
    if not os.path.exists(p.confusion_dir):
        os.mkdir(p.confusion_dir)
    Plops.make_montage(tp, title='True Positives', size=p.BATCH_SIZE)
    Plops.make_montage(tn, title='True Negatives', size=p.BATCH_SIZE)
    Plops.make_montage(fp, title='False Positives', size=p.BATCH_SIZE)
    Plops.make_montage(fn, title='False Negatives', size=p.BATCH_SIZE)


if ID_RESULTS:
    if not os.path.exists(p.res_csv_deploy):
        os.mkdir(p.res_csv_deploy)
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(save_res)

print('tp', np.shape(tp))
print('tn', len(tn))
print('fp', len(fp))
print('fn', len(fn))
test_accuracy = np.mean(test_accuracy_lst)
print('test accuracy', test_accuracy)
