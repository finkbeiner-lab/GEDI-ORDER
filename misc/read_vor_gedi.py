"""
Get files from csv with gedi info for one of the dream datasets and make TFrecord from list
"""

import pandas as pd
import os
import glob
import numpy as np
from preprocessing.create_tfrecs_from_lst import Record
import random
import param_gedi as param


def append_files(df, lst):
    cnt = 0
    for i, row in df.iterrows():
        cnt += 1
        exp = row.Plate
        w = row.Well
        ident = row.Object
        tp = int(row.Timepoint)
        f = glob.glob(os.path.join('/mnt/data/MATCHING/VORONOI/CorrectedCentroidCrops', exp, w,
                                   '*_{}_T{}_*_{}_*_{}.tif'.format(exp, tp, w, ident)))
        if len(f) > 0:
            assert len(f) == 1, 'multiple files'
            f = f[0]
            lst.append(f)


p = param.Param()
FROM_NPZ = True
parent = '/mnt/data/GEDI-ORDER/GEDI-DATA'
name = 'ForJosh_GEDIbiosensor_HumanCuration.csv'
f = os.path.join(parent, name)
sv = os.path.join(parent, 'GEDIbiosensor_filelist.npz')
identifier = 'GEDIbiosensor'

if not FROM_NPZ:

    df = pd.read_csv(f)
    df = df[~df.Timepoint.isna()]

    livedf = df[df.GEDI == 1]
    deaddf = df[df.GEDI == 0]
    livefiles = []
    deadfiles = []
    append_files(livedf, livefiles)
    append_files(deaddf, deadfiles)

    np.savez(sv, live=livefiles, dead=deadfiles)

else:
    arr = np.load(sv)
    livefiles = list(arr['live'])
    deadfiles = list(arr['dead'])
    length = len(deadfiles)
    random.seed(1)
    livepaths = random.sample(livefiles, length)
    deadpaths = deadfiles
    split = [.7, .15, .15]
    Rec = Record(livepaths, deadpaths, p.tfrecord_dir, split, scramble=False)
    savetrain = os.path.join(p.tfrecord_dir, 'vor_{}_train.tfrecord'.format(identifier))
    saveval = os.path.join(p.tfrecord_dir, 'vor_{}_val.tfrecord'.format(identifier))
    savetest = os.path.join(p.tfrecord_dir, 'vor_{}_test.tfrecord'.format(identifier))
    print(Rec.trainlbls)
    Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
    Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
    Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
