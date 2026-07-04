from train import Train

if __name__ == '__main__':
    result = pyfiglet.figlet_format("GEDI-CNN", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Train binary classifer GEDI-CNN model')
    # positives = ['/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Livecrops_3',
    #              '/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Livecrops_2',
    #              '/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Livecrops_1']
    # negatives = ['/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Deadcrops_3',
    #              '/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Deadcrops_2',
    #              '/mnt/finkbeinernas/robodata/Shijie/ML/NSCLC-H23/Deadcrops_1']

    positives = ['/gladstone/finkbeiner/linsley/Shijie/ML/NSCLC-1703/Livecrops_1',
                 '/gladstone/finkbeiner/linsley/Shijie/ML/NSCLC-1703/Livecrops_2_3']
    negatives = ['/gladstone/finkbeiner/linsley/Shijie/ML/NSCLC-1703/Deadcrops_1',
                 '/gladstone/finkbeiner/linsley/Shijie/ML/NSCLC-1703/Deadcrops_2_3']
    parser.add_argument('--datadir', action="store", nargs='+',
                        default='/gladstone/finkbeiner/linsley/Josh/GEDI-ORDER',
                        help='data parent directory',
                        dest='datadir')
    parser.add_argument('--res_dir', action="store",
                        default='/gladstone/finkbeiner/linsley/Josh/GEDI-ORDER',
                        help='data parent directory',
                        dest='res_dir')
    parser.add_argument('--pos_dir', nargs='+',
                        default=positives,
                        help='directory with positive images', dest="pos_dir")
    parser.add_argument('--neg_dir', nargs='+',
                        default=negatives,
                        help='directory with negative images', dest="neg_dir")
    parser.add_argument('--balance_method', action="store", default='multiply',
                        help='method to handle unbalanced data: cutoff, multiply or none', dest="balance_method")
    parser.add_argument('--preprocess_tfrecs', type=int, action="store", default=False,
                        help='generate tfrecords, necessary for new datasets, if already generate set to false',
                        dest="preprocess_tfrecs")
    parser.add_argument('--use_wandb', type=int, action="store", default=True,
                        help='Save run info to neptune ai',
                        dest="use_wandb")
    parser.add_argument('--retrain', type=int, action="store", default=False,
                        help='Save run info to neptune ai',
                        dest="retrain")
    args = parser.parse_args()
    print('ARGS:\n', args)
    for datadir, pos_dir in zip(args.datadir, args.pos_dir):
        Tr = Train(parent_dir=args.datadir, res_dir=args.res_dir, param_dict=None,
                   preprocess_tfrecs=args.preprocess_tfrecs,
                   use_wandb=args.use_wandb)
        Tr.run(pos_dir, args.neg_dir, args.balance_method)
