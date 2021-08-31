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
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

print('Running...')
# Setup filepaths and csv to log info about training model
p = param.Param()
make_directories(p)
timestamp = update_timestring()
export_path = os.path.join(p.models_dir, '{}_{}.h5'.format(p.which_model, timestamp))
export_info_path = os.path.join(p.run_info_dir, '{}_{}.csv'.format(p.which_model, timestamp))
save_checkpoint_path = os.path.join(p.ckpt_dir, '{}_{}.hdf5'.format(p.which_model, timestamp))
run_info = {'model': p.which_model,
            'retraining': '',
            'timestamp': timestamp,
            'model_timestamp': p.which_model + '_' + timestamp,
            'train_path': p.data_train,
            'val_path': p.data_val,
            'test_path': p.data_test,
            'learning_rate': p.learning_rate,
            'augmentation': p.augmentbool,
            'epochs': p.EPOCHS,
            'batch_size': p.BATCH_SIZE,
            'output_size': p.output_size,
            'im_shape': p.target_size,
            'random_crop': p.randomcrop}
print(run_info)


net = CNN()
if p.which_model == 'vgg16':
    model = net.vgg16(imsize=p.target_size)
    input_name = 'input_1'
elif p.which_model == 'vgg19':
    model = net.vgg19(imsize=p.target_size)
    input_name = 'vgg19_input'
elif p.which_model == 'mobilenet':
    model = net.mobilenet(imsize=p.target_size)
elif p.which_model == 'inceptionv3':
    model = net.inceptionv3(imsize=p.target_size)
    input_name = 'inception_v3_input'
else:
    model = net.standard_model(imsize=p.target_size)


# Get length of tfrecords
Chk = pipe.Dataspring(p.data_train)
train_length = Chk.count_data().numpy()
del Chk
Chk = pipe.Dataspring(p.data_val)
val_length = Chk.count_data().numpy()
del Chk
Chk = pipe.Dataspring(p.data_test)
test_length = Chk.count_data().numpy()
del Chk
DatTrain = pipe.Dataspring(p.data_train)
DatVal = pipe.Dataspring(p.data_val)
DatTest = pipe.Dataspring(p.data_test)
DatTest2 = pipe.Dataspring(p.data_test)

train_ds = DatTrain.datagen_base(istraining=True)
val_ds = DatVal.datagen_base(istraining=True)
test_ds = DatTest.datagen_base(istraining=False)
test_ds2 = DatTest2.datagen_base(istraining=False)
print('training length', train_length)
print('validation length', val_length)
print('test length', test_length)
train_gen = DatTrain.generator(input_name)
val_gen = DatVal.generator(input_name)
test_gen = DatTest.generator(input_name)


# callbacks, save checkpoints and tensorboard logs
cp_callback = tf.keras.callbacks.ModelCheckpoint(save_checkpoint_path, monitor='val_accuracy', verbose=1,
                                                 save_best_only=True, mode='max')

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(p.tb_log_dir, p.which_model),
    update_freq='epoch')

run = neptune.init(project='stephanie.lam/GEDI-ORDER-DNA',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMjY2YmJkYi01N2RlLTQxNTItYWVhMS01NTljMTg4OGU3MjAifQ==') # your credentials
run['run_info'] = run_info
neptune_callback = NeptuneCallback(run=run, base_namespace='metrics')

callbacks = [cp_callback, tb_callback, neptune_callback]
history = model.fit(train_gen, steps_per_epoch=train_length // (p.BATCH_SIZE), epochs=p.EPOCHS,
                    class_weight=p.class_weights, validation_data=val_gen,
                    validation_steps=val_length // p.BATCH_SIZE, callbacks=callbacks, workers=4,
                    use_multiprocessing=True)

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
# Freeze whole model for evaluation
# for lyr in model.layers:
#     lyr.trainable = False
# model.trainable = False

# Predict on test dataset
res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE, workers=4, use_multiprocessing=True)
test_accuracy_lst = []
# Get accuracy, compare predictions with labels
for i in range(int(test_length // p.BATCH_SIZE)):
    imgs, lbls, files = DatTest2.datagen()

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

run["train/accuracy"].log(train_acc[-1])
run["train/loss"].log(train_loss[-1])
run['test/accuracy'].log(test_accuracy)
run['val/accuracy'].log(val_acc[-1])
run['val/loss'].log(val_loss[-1])

run_info['train_accuracy'] = train_acc[-1]
run_info['val_accuracy'] = val_acc[-1]
run_info['test_accuracy'] = test_accuracy
run_info['train_loss'] = train_loss[-1]
run_info['val_loss'] = val_loss[-1]

run_df = pd.DataFrame([run_info])
run_df.to_csv(export_info_path)

print('Saving model to {}'.format(export_path))
model.save(export_path)
