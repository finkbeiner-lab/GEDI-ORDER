"""Loop through hyperparameters and train"""
from main.train_from_file import Train
import argparse
import pyfiglet


def gridsearch(datadir):
    param_dict = {'epochs': 200, 'augmentbool': True, 'batch_size': 64, 'min_contrast': 1, 'max_contrast': 1.3,
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
                Tr = Train(parent_dir=datadir, param_dict=param_dict, preprocess_tfrecs=False, use_neptune=True)
                Tr.train()


if __name__ == '__main__':
    result = pyfiglet.figlet_format("GEDI-CNN", font="slant")
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', action="store",
                        default='/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER',
                        help='data parent directory',
                        dest='datadir')
    args = parser.parse_args()
    print('ARGS:\n', args)
    gridsearch(args.datadir)
