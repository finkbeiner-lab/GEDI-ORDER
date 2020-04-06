import tensorflow as tf
import param_gedi as param
from models.model import CNN
import preprocessing.datagenerator as pipe
from utils.utils import update_timestring, make_directories
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

p = param.Param()
make_directories(p)
timestamp = update_timestring()
export_path = os.path.join(p.models_dir, '{}_{}.h5'.format(p.which_model, timestamp))
export_info_path = os.path.join(p.run_info_dir, '{}_{}.csv'.format(p.which_model, timestamp))
save_checkpoint_path = os.path.join(p.ckpt_dir, '{}_{}.hdf5'.format(p.which_model, timestamp))
run_info = {'model': p.which_model,
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
            'im_shape': p.target_size}

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

train_ds = DatTrain.datagen_base(istraining=True)
val_ds = DatVal.datagen_base(istraining=True)
test_ds = DatTest.datagen_base(istraining=False)
print('training length', train_length)
print('validation length', val_length)
print('test length', test_length)
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

# callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(save_checkpoint_path, monitor='val_accuracy', verbose=1,
                                                 save_best_only=True, mode='max')

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='/home/jlamstein/PycharmProjects/ASYN/log/{}'.format(p.which_model),
    update_freq='epoch')

callbacks = [tb_callback, cp_callback]
history = model.fit(train_gen, steps_per_epoch=train_length // (p.BATCH_SIZE), epochs=p.EPOCHS,
                    class_weight=p.class_weights, validation_data=val_gen,
                    validation_steps=val_length // p.BATCH_SIZE)

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
# Freeze model, especially necessary because I use dropout
for lyr in model.layers:
    lyr.trainable = False

res = model.predict(test_gen, steps=test_length // p.BATCH_SIZE)
test_accuracy_lst = []
for i in range(int(test_length // p.BATCH_SIZE)):
    imgs, lbls, files = DatTest.datagen()
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
