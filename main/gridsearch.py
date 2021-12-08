"""Loop through hyperparameters and train"""
from main.train_from_file import Train


def gridsearch():
    datadir = '/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER'
    param_dict = {'epochs': 200, 'augmentbool': True, 'min_contrast': 1, 'max_contrast': 1.3,
                  'random_brightness': .2, 'target_size': (224, 224, 3), 'orig_size': (300, 300, 1),
                  'class_weights': {0: 1., 1: 1.}, 'momentum': .9, 'randomcrop': True}
    models = ['vgg19', 'resnet50']
    lrs = [1e-6, 1e-5]
    optimizers = ['sgd', 'adam']

    for model in models:
        for lr in lrs:
            for optimizer in optimizers:
                param_dict['model'] = model
                param_dict['learning_rate'] = lr
                param_dict['optimizer'] = optimizer
                Train(parent_dir=datadir, param_dict=param_dict, preprocess_tfrecs=False, use_neptune=True)
