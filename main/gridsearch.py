"""Loop through hyperparameters and train"""
from main.train import Train
import argparse
import pyfiglet


def gridsearch(datadir, retrain_bool):
    param_dict = {'epochs': 200, 'augmentbool': True, 'batch_size': 32, 'min_contrast': 1, 'max_contrast': 1.3,
                  'random_brightness': .2, 'target_size': (224, 224, 3), 'orig_size': (300, 300, 1),
                  'class_weights': {0: 1., 1: 1.}, 'momentum': .9, 'randomcrop': True, 'histogram_eq': True}
    models = ['vgg19']
    batch_sizes = [32, 64]
    lrs = [1e-4, 1e-5, 1e-6]
    optimizers = ['sgd', 'adam', 'adamw']
    l2s = [0]
    wds = [1e-5]
    momentums = [.9]
    regs = [None]
    for model in models:
        for batch_size in batch_sizes:
            for wd in wds:
                for lr in lrs:
                    for optimizer in optimizers:
                        for l2 in l2s:
                            for reg in regs:
                                for momentum in momentums:
                                    param_dict['batch_size']= batch_size
                                    param_dict['model'] = model
                                    param_dict['learning_rate'] = lr
                                    param_dict['optimizer'] = optimizer
                                    param_dict['regularize'] = reg
                                    param_dict['l2_regularize'] = l2
                                    param_dict['weight_decay'] = wd
                                    param_dict['momentum'] = momentum
                                    Tr = Train(parent_dir=datadir, param_dict=param_dict, preprocess_tfrecs=False,
                                               use_neptune=True)
                                    if retrain_bool:
                                        Tr.retrain()
                                    else:
                                        Tr.train()


if __name__ == '__main__':
    result = pyfiglet.figlet_format("GEDI-CNN", font="slant")
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', action="store",
                        default='/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER',
                        help='data parent directory',
                        dest='datadir')
    parser.add_argument('--retrain', type=int, action="store", default=False,
                        help='Save run info to neptune ai',
                        dest="retrain")
    args = parser.parse_args()
    print('ARGS:\n', args)
    gridsearch(args.datadir, args.retrain)
