"""Loop through hyperparameters and train"""
from main.train import Train
import argparse
import pyfiglet


def gridsearch(datadir, res_dir, retrain_bool):
    param_dict = {'epochs': 100, 'augmentbool': True, 'batch_size': 32, 'min_contrast': 1, 'max_contrast': 1.3,
                  'random_brightness': .2, 'target_size': (224, 224, 3), 'orig_size': (300, 300, 1),
                  'class_weights': {0: 1., 1: 1.}, 'momentum': .9, 'randomcrop': True, 'histogram_eq': True}
    models = ['vgg19']
    batch_sizes = [32]
    lrs = [5e-5, 1e-5, 5e-6]
    optimizers = ['adam', 'sgd', 'adamw']
    l2s = [0]
    wds = [1e-5]
    momentums = [.9]
    regs = [None]
    brightnesses = [.1, .2, .3]
    max_contrasts = [1.1, 1.3, 1.5]
    for model in models:
        for batch_size in batch_sizes:
            for wd in wds:
                for lr in lrs:
                    for optimizer in optimizers:
                        for l2 in l2s:
                            for reg in regs:
                                for momentum in momentums:
                                    for brightness in brightnesses:
                                        for max_contrast in max_contrasts:
                                            param_dict['batch_size']= batch_size
                                            param_dict['model'] = model
                                            param_dict['learning_rate'] = lr
                                            param_dict['optimizer'] = optimizer
                                            param_dict['regularize'] = reg
                                            param_dict['l2_regularize'] = l2
                                            param_dict['weight_decay'] = wd
                                            param_dict['momentum'] = momentum
                                            param_dict['random_brightness'] = brightness
                                            param_dict['max_contrast'] = max_contrast
                                            Tr = Train(parent_dir=datadir,res_dir=res_dir, param_dict=param_dict, preprocess_tfrecs=False,
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
    parser.add_argument('--res_dir', action="store",
                        default='/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER',
                        help='data parent directory',
                        dest='res_dir')
    parser.add_argument('--retrain', type=int, action="store", default=False,
                        help='Save run info to neptune ai',
                        dest="retrain")
    args = parser.parse_args()
    print('ARGS:\n', args)
    gridsearch(args.datadir, args.res_dir, args.retrain)
