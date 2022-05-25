"""Move or delete this file
Combine tracking csvs with gedi ratio csvs
Group cells into time
Rename crops based on voronoi tracking
Move crops to live/dead folder
Train"""

import pandas as pd
import imageio
from glob import glob
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt


# Combine tracking csvs with gedi ratio csvs
# pd merge
class MergeTrackingWithGedi:
    def __init__(self, tracking_csv, gedi_csv, gedi_csv2, condition=None):
        self.track = pd.read_csv(tracking_csv)
        self.track = self.track.loc[:, ~self.track.columns.str.contains('^Unnamed')]
        self.gedi = pd.read_csv(gedi_csv)
        self.gedi = self.gedi.loc[:, ~self.gedi.columns.str.contains('^Unnamed')]
        self.gedi = self.gedi[['Sci_WellID', 'Sci_PlateID', 'ObjectLabelsFound', 'Timepoint',
                               'Confocal_GFPPixelIntensityMean', 'GEDI', 'GEDI_confocal', 'GEDILabel']]
        self.savepath = tracking_csv.split('.c')[0] + '_GEDI.csv'
        self.condition1 = condition[0]
        self.gedi['line'] = self.condition1
        if gedi_csv2 is None:
            self.gedi_csv2 = None
            self.condition2 = None
        else:
            self.gedi2 = pd.read_csv(gedi_csv2)
            self.gedi2 = self.gedi2.loc[:, ~self.gedi2.columns.str.contains('^Unnamed')]
            self.gedi2 = self.gedi2[['Sci_WellID', 'Sci_PlateID', 'ObjectLabelsFound', 'Timepoint',
                                     'Confocal_GFPPixelIntensityMean', 'GEDI', 'GEDI_confocal', 'GEDILabel']]
            self.condition2 = condition[1]
            self.gedi2['line'] = self.condition2
            self.gedi = pd.concat([self.gedi, self.gedi2])

    def run(self):
        # merge on multiple cols
        print('track len', len(self.track))
        print('gedi len', len(self.gedi))
        df = pd.merge(self.track, self.gedi, how='inner',
                      on=['Sci_WellID', 'Sci_PlateID', 'ObjectLabelsFound', 'Timepoint'])

        print('merged length', len(df))
        df.to_csv(self.savepath)


# Group cells into time
# Rename crops based on voronoi tracking

class RenameCropsWithTrackID:
    def __init__(self, f, exp, parent_dir):
        self.f = f
        self.exp = exp
        self.parent_dir = parent_dir
        self.savecropdir = os.path.join(self.parent_dir, 'ObjectCropsVoronoi')

    def run(self):
        self.filterdf(self.f)
        self.copyCrop()

        #

    def filterdf(self, file):
        img = imageio.imread(glob(os.path.join(self.parent_dir, 'AlignedImages', '**', '*.tif'))[0])
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]

        self.df = pd.read_csv(file, low_memory=False)
        self.df = self.df[self.df.MeasurementTag.str.contains('Epi-GFP16')]  # crops are triples with multiple channels
        self.NG = NittyGritty(self.image_width, self.image_height)
        # self.df, typo_indices = self.NG.cut_off_typo_tracks(self.df)
        self.df = self.NG.remove_neg_coords(self.df)
        # self.df = self.NG.remove_untracked(self.df)
        self.df, broken_indices = self.NG.cut_off_broken_tracks(self.df)
        self.df = self.NG.remove_singles(self.df)

    def copyCrop(self):
        for i, row in self.df.iterrows():
            well = row.Sci_WellID
            plate = row.Sci_PlateID
            galaxy_lbl = row.ObjectLabelsFound
            tp = row.Timepoint
            vor_lbl = row.graph_id
            crops = glob(os.path.join(self.parent_dir, 'ObjectCrops', well,
                                      f'*_T{tp}_*_{row.Sci_WellID}_*_{galaxy_lbl}.tif'))
            for crop in crops:
                savedir = os.path.join(self.savecropdir, well)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                tokens = crop.split('/')[-1].split('.t')[0].split('_')
                vorcropname = '_'.join(tokens[:-1]) + f'_{vor_lbl}.tif'
                vorcrop = os.path.join(savedir, vorcropname)
                shutil.copyfile(crop, vorcrop)


class PrepareForSoothsayer:
    def __init__(self, f, parent_dir, exp):
        self.df = pd.read_csv(f)
        self.exp = exp
        self.cropdir = os.path.join(parent_dir, 'ObjectCropsVoronoi')
        self.savedir = os.path.join(parent_dir, 'SOOTH_DATA')
        self.parent_dir = parent_dir
        img = imageio.imread(glob(os.path.join(self.parent_dir, 'AlignedImages', '**', '*.tif'))[0])
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        self.df = self.df[self.df.MeasurementTag.str.contains('Epi-GFP16')]  # crops are triples with multiple channels
        self.NG = NittyGritty(self.image_width, self.image_height)
        # self.df, typo_indices = self.NG.cut_off_typo_tracks(self.df)
        self.df = self.NG.remove_neg_coords(self.df)
        # self.df = self.NG.remove_untracked(self.df)
        self.df, broken_indices = self.NG.cut_off_broken_tracks(self.df)
        self.df = self.NG.remove_singles(self.df)
        self.df = self.df[self.df.GEDILabel != 'Discard']

    def sortClasses(self):
        labels = self.df.line.tolist()
        unis = np.unique(labels)
        dfs = []
        classkey = {'Live': 0, 'Dead': 2, 'Discard': 1}
        after_discard = {'Live': 0, 'Dead': 0, 'Discard': 0}
        zombie = []  # either tracking error or cell moving
        good = []
        # create dirs
        # for lbl in unis:
        #     savepath = os.path.join(self.savedir, lbl)  # directories based on line
        #     if not os.path.join(savepath):
        #         os.makedirs(savepath)

        g = self.df.sort_values('Timepoint', ascending=True).groupby(['Sci_WellID', 'graph_id'])
        # print(g.head)

        # g = g.apply(lambda x: x.GEDILabel.shift(periods=1))
        plt.figure()
        cnt = 0

        for name, df in g:
            classes = df.GEDILabel.tolist()
            y = []
            for cls in classes:
                y.append(classkey[cls])
            y = np.array(y) + np.random.random(len(y)) / 10
            x = df.Timepoint.to_numpy() + np.random.random(len(y)) / 10

            if 'Dead' in df.GEDILabel.tolist():
                flag = False
                goodFlag = True
                for cls in classes:
                    if flag:
                        if cls=='Live':
                            zombie.append(name)
                            goodFlag = False
                    if cls == 'Dead':
                        flag = True
                    else:
                        flag = False
                if goodFlag:
                    good.append(name)
                cnt += 1
                # plt.plot(x, y)
            shfted = classes[1:] + ["Null"]
            df.GEDILabel = shfted
            dfs.append(df)

            # print(name)
            # print(df.GEDILabel)
            # print(g.head)
            # if not cnt % 10:
            #     plt.show()
        # print('after discard', after_discard)
        print('number of tracks', len(g))

        print('zombies from mistracks or cancer cells interacting', len(zombie), zombie)
        print('good', len(good))
        newdf = pd.concat(dfs)
        newdf = newdf[newdf.GEDILabel!='Null']
        newdf = newdf[newdf.Timepoint > 0]
        gg = newdf.sort_values('Timepoint', ascending=True).groupby(['Sci_WellID', 'graph_id'])

        for name, df in gg:
            classes = df.GEDILabel.tolist()
            print(name)
            # print(classes)

            for i, row in df.iterrows():
                well = row.Sci_WellID
                plate = row.Sci_PlateID
                tp = row.Timepoint
                neuron_id = row.graph_id
                crops = glob(os.path.join(self.parent_dir, 'ObjectCropsVoronoi', well,
                                          f'*_T{tp}_*_{row.Sci_WellID}_*_Confocal-GFP16_*_{neuron_id}.tif'))
                for crop in crops:
                    savefolder = os.path.join(self.savedir, row.GEDILabel)
                    if not os.path.exists(savefolder):
                        os.makedirs(savefolder)
                    name = crop.split('/')[-1]
                    savepath = os.path.join(savefolder, name)
                    shutil.copyfile(crop, savepath)

        # group and sort to get timepoint chains, rotate the label t-1
        # split and copy
        # balance method handled in train.py


class NittyGritty:
    def __init__(self, w, h):
        self.image_height = h
        self.image_width = w

    def remove_untracked(self, df):
        res = df.dropna(subset=['Live_Cells'])
        return res

    def remove_singles(self, df):
        df['nObjects'] = df.groupby(["Sci_WellID", 'graph_id'])['Sci_PlateID'].transform('count')
        df = df[df.nObjects > 1]
        df = df.drop(columns=['nObjects'])
        return df

    def starts_at_zero(self, data):
        data['start_at_zero'] = True
        grp = data.groupby(by=['Sci_WellID', 'graph_id'])
        for name, df in grp:
            tps = df.Timepoint.values
            start_at_zero = np.min(tps) == 0
            if not start_at_zero:
                data.loc[(data.Sci_WellID == name[0]) & (data.ObjectTrackID == name[1]), 'start_at_zero'] = False
        res = data[data.start_at_zero == True]
        res = res.drop(columns=['start_at_zero'])
        return res

    def remove_neg_coords(self, df):
        wrong = df[(df.BlobCentroidX < 0) | (df.BlobCentroidY < 0) |
                   (df.BlobCentroidX > self.image_width) | (df.BlobCentroidY > self.image_height)]
        df = df[(df.BlobCentroidX >= 0) & (df.BlobCentroidY >= 0) &
                (df.BlobCentroidX < self.image_width) & (df.BlobCentroidY < self.image_height)]
        wrong_tracks = wrong.loc[wrong.Timepoint == 0, ['Sci_WellID', 'ObjectTrackID']]

        df = df[(df.BlobCentroidX >= 0) & (df.BlobCentroidY >= 0) &
                (df.BlobCentroidX < self.image_width) & (df.BlobCentroidY < self.image_height)]
        for i, row in wrong_tracks.iterrows():
            df = df.drop(df[(df.Sci_WellID == row[0]) & (df.ObjectTrackID == row[1])].index)
        return df

    # def remove_untracked_with_master(self, data, master):
    #     master_grp = master.groupby(by=['Well', 'ObjectTrackID'])
    #     grp = data.groupby(by=['Sci_WellID', 'ObjectTrackID'])
    #     for name, df in grp:
    #         for i, row in df.iterrows():

    @staticmethod
    def cut_off_broken_tracks(data):
        broken_indices = []
        grp = data.groupby(by=['Sci_WellID', 'graph_id'])
        for name, df in grp:

            tps = df.Timepoint.values
            start_at_zero = np.min(tps) == 0

            if len(tps) > 1 and start_at_zero:
                tps = [i for i in tps if i > 0]
                now_start_at_one = np.min(tps) == 1
                if not now_start_at_one:
                    idxs = df.index
                    broken_indices.extend(idxs)
                    continue
                total_len = len(df)
                expected_tps = [i for i in range(total_len) if i > 0]
                flag = np.prod(tps) == np.prod(expected_tps)
                if not flag:
                    df_srt = df.sort_values(by='Timepoint')
                    cnt = -1
                    broken_tp = None
                    for idx, row in df_srt.iterrows():
                        if cnt > 0:
                            dt = row.Timepoint - prev_row.Timepoint
                            if dt != 1:
                                broken_tp = row.Timepoint
                                break
                        prev_row = row
                        cnt += 1
                    if broken_tp is not None:
                        broke_df = df[df.Timepoint >= broken_tp]
                        idxs = broke_df.index
                        broken_indices.extend(idxs)
        res = data.drop(broken_indices)
        print(f'Found {len(broken_indices)} broken timepoints\n cutting off tracks...')
        return res, broken_indices


if __name__ == '__main__':
    tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/Voronoi/2022_05_03_17_51_02_cell_data_TRACKED.csv'
    gedi_tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/Voronoi/2022_05_03_17_51_02_cell_data_TRACKED_GEDI.csv'
    parent_dir = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM'
    exp = '2022-0429-NSCLC-H23-10uM'
    image_dir = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/AlignedImages'
    gedi_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/H23-10uM_GEDI.csv'
    gedi_csv2 = None
    condition = ['H23']

    # tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-1703-10uM/Voronoi/2022_05_03_18_10_39_cell_data_TRACKED.csv'
    # gedi_tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-1703-10uM/Voronoi/2022_05_03_18_10_39_cell_data_TRACKED_GEDI.csv'
    # image_dir = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/AlignedImages'
    # parent_dir = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-1703-10uM/'
    # exp = '2022-0429-NSCLC-H23-10uM'
    # gedi_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-1703-10uM/1703-10uM_GEDI.csv'
    # gedi_csv2 = None
    # condition = ['1703']

    # tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM/Voronoi/2022_05_04_10_36_31_cell_data_TRACKED.csv'
    # gedi_tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM/Voronoi/2022_05_04_10_36_31_cell_data_TRACKED_GEDI.csv'
    # parent_dir = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM'
    # exp = '2022-0429-NSCLC-10uM'
    # gedi_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM/NSCLC-half-1703-10uM_GEDI.csv'
    # gedi_csv2 = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM/NSCLC-half-H23-10uM_GEDI.csv'
    # condition = ['1703', 'H23']

    print(exp)
    # MRG = MergeTrackingWithGedi(tracking_csv, gedi_csv, gedi_csv2, condition)
    # MRG.run()

    # Ren = RenameCropsWithTrackID(gedi_tracking_csv, exp, parent_dir)
    # Ren.run()

    Prp = PrepareForSoothsayer(gedi_tracking_csv, parent_dir, exp)
    Prp.sortClasses()
