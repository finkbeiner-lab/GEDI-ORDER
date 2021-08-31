"""
Generate tfrecords for dream challenge transfer learning.
LINCS062016A max inten 2000-5000, processed with pilot
LINCS062016B low contrast, max < 1000
LINCS092016A low contrast, max < 1000
LINCS092016B low contrast, max < 1000

"""

import pandas as pd
import glob
import os
from preprocessing.create_tfrecs_from_lst import Record
import param_gedi as param

p = param.Param()

parent = '/mnt/data/MATCHING/DREAM_DATA/Jeremy'
cropdir = '/mnt/data/MATCHING/VORONOI/CorrectedCentroidCrops'
experiments = ['LINCS092016A']
# experiments = ['LINCS062016B', 'LINCS092016A', 'LINCS092016B']
cnt_lim = 1e16
identifier = ''
for _e in experiments:
    identifier += _e + '|'
identifier = identifier[:-1]
livefile = os.path.join(parent, 'Dream_Live.csv')
deadfile = os.path.join(parent, 'Dream_Dead.csv')
live_df = pd.read_csv(livefile)
dead_df = pd.read_csv(deadfile)
live_df = live_df[live_df.Experiment.str.contains(identifier)]
dead_df = dead_df[dead_df.Experiment.str.contains(identifier)]

livepaths = []
deadpaths = []
split = [.7, .15, .15]


def append_files(df, lst):
    cnt = 0
    for i, row in df.iterrows():
        cnt += 1
        exp = row.Experiment
        w = row.Well
        ident = row.ObjectTrackID
        tp = row.TimePoint
        f = glob.glob(os.path.join(cropdir, exp, w, '*_{}_T{}_*_{}_*_{}.tif'.format(exp, tp, w, ident)))
        if len(f) > 0:
            assert len(f) == 1, 'multiple files'
            f = f[0]
            lst.append(f)
        # if cnt > cnt_lim:
        #     break


print('Collecting files from csv')
append_files(live_df, livepaths)
append_files(dead_df, deadpaths)
print('Generating tfrecords...')
idx = 0
deadlen = len(deadpaths)
newdeadlen = deadlen
print('duplicating dead paths to match number of live paths')
while newdeadlen < len(livepaths):
    deadpaths += [deadpaths[idx % deadlen]]
    idx += 1
    assert len(deadpaths) == newdeadlen + 1
    newdeadlen = len(deadpaths)

Rec = Record(livepaths, deadpaths, p.tfrecord_dir, split, scramble=False)

savetrain = os.path.join(p.tfrecord_dir, 'vor_{}_train.tfrecord'.format(identifier))
saveval = os.path.join(p.tfrecord_dir, 'vor_{}_val.tfrecord'.format(identifier))
savetest = os.path.join(p.tfrecord_dir, 'vor_{}_test.tfrecord'.format(identifier))
Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
