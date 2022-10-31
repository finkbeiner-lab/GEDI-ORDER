"""Add gedi results from csv to tracking csv (cell_data.csv) from galaxy runs.
Sort by track and timepoints.
Report live / dead over time"""

import pandas as pd
import numpy as np
import os
import pyfiglet
import argparse
from glob import glob
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, CoxTimeVaryingFitter
from string import ascii_uppercase
from lifelines.statistics import multivariate_logrank_test, logrank_test
import imageio
import random
import cv2
from tqdm import tqdm


def starts_at_zero(data):
    data['start_at_zero'] = True
    grp = data.groupby(by=['Sci_WellID', 'ObjectLabelsFound'])
    for name, df in grp:
        tps = df.Timepoint.values
        start_at_zero = np.min(tps) == 0
        if not start_at_zero:
            data.loc[(data.Sci_WellID == name[0]) & (data.ObjectTrackID == name[1]), 'start_at_zero'] = False
    res = data[data.start_at_zero == True]
    res = res.drop(columns=['start_at_zero'])
    return res


class trackSurvival:
    def __init__(self):
        self.label_dict = {}

    def merge_dfs(self, gedicsv, trackcsv, savename):
        # filepath, prediction
        # /media/data/GEDI/drew_images/original_images/Live_all_rh_analysis_rat_train/bs_cache_RGEDIMachineLearning25_B3_52_FITC-DFTrCy5_RFP-DFTrCy5.tif
        # PID20220615_iMG-coculture-1-061522_T1_0-1_A04_0_FITC_0_1_0_BGs_MN_ALIGNED_1.tif
        gedi = pd.read_csv(gedicsv, low_memory=False)
        track = pd.read_csv(trackcsv, low_memory=False)
        filenames = gedi.filepath.str.split('/').str[-1]
        tmp = filenames.str.split('_', expand=True)
        gedi['Sci_PlateID'] = tmp[1]
        gedi['Timepoint'] = tmp[2].str.slice(start=1).astype(int)
        gedi['Sci_WellID'] = tmp[4]
        gedi['ObjectLabelsFound'] = tmp[len(tmp.columns)-1].str.split('.').str[0].astype(int)
        gedi = gedi.rename(columns={'prediction': 'gedi'})
        gedi['MeasurementTag'] = tmp[6]
        gedi = gedi.drop(columns=['filepath', 'label'])
        track['Sci_PlateID'] = track['Sci_PlateID'].str.split('_').str[1]
        gedi = gedi.loc[:, ~gedi.columns.str.contains('^Unnamed')]
        track = track.loc[:, ~track.columns.str.contains('^Unnamed')]
        merged = pd.merge(track, gedi)
        # ObjectLabelsFound
        # Sci_PlateID
        # Sci_WellID
        # MeasurementTag
        # Timepoint
        merged.to_csv(savename, index=False)
        print(f'Saved to {savename}')

    def death_check(self, df):
        dead_flag = False
        time_of_death = -1
        for i, row in df.iterrows():
            if row['gedi']:
                if dead_flag:
                    return time_of_death, True
            elif not row['gedi']:
                if time_of_death == -1:
                    time_of_death = row['Timepoint']
                dead_flag = True
        return time_of_death, False

    def live_dead_tracks(self, gedi_cell_data, platemapcsv):
        use_samples = True
        data = pd.read_csv(gedi_cell_data, low_memory=False)
        data = data.sort_values(by='Timepoint')
        platemap = pd.read_csv(platemapcsv)
        str_labels = platemap.Sci_SampleID.unique()
        for i, lbl in enumerate(str_labels):
            self.label_dict[i] = lbl
        platemap['label'] = -1
        for i, lbl in self.label_dict.items():
            platemap.loc[platemap.Sci_SampleID == lbl, 'label'] = i

        plate_groups = platemap.groupby(by='label')
        zombies = []
        death_dict = {}
        zombie_dict = {}
        death_dfs = []
        death_lst = []
        regression_dfs = []
        data['label'] = ''
        data['condition'] = ''
        max_timepoint = int(data.Timepoint.max())
        min_timepoint = int(data.Timepoint.min())

        if use_samples:
            for SampleID, grp in plate_groups:
                wells = grp.Sci_WellID
                for w in wells:
                    data.loc[data.Sci_WellID == w, 'label'] = SampleID
                    data.loc[data.Sci_WellID == w, 'condition'] = self.label_dict[SampleID]
        # else:
        #     for SampleID in range(1, 30):  # todo: redundant
        #         wells = [ascii_uppercase[x] + str(SampleID).zfill(2) for x in range(26)]
        #         for w in wells:
        #             data.loc[data.Sci_WellID == w, 'label'] = SampleID
        #             data.loc[data.Sci_WellID == w, 'condition'] = self.label_dict[SampleID]


        filt_df = data.groupby(by=['Sci_WellID', 'ObjectLabelsFound']).filter(lambda x: min(x['Timepoint']) == min_timepoint)
        # only take groups that start with T1, no T0 in this dataset
        groups = filt_df.groupby(by=['Sci_WellID', 'ObjectLabelsFound'])

        gedi_sum = groups['gedi'].sum()
        all_dead = gedi_sum[gedi_sum == 0]
        # all_live_df = groups.filter(lambda x: x.gedi.all())
        # all_dead_df = groups.filter(lambda x: x.gedi.sum()==0)
        gedi_all = groups['gedi'].all()
        all_live = gedi_all[gedi_all]
        for name, cell in groups:
            if name not in all_dead and name not in all_live:
                time_of_death, zombie_bool = self.death_check(cell)
                if zombie_bool:
                    zombies.append(cell)
                    zombie_dict[name] = time_of_death
                else:
                    death_lst.append(time_of_death)
                    death_dict[name] = [time_of_death, cell]
                    death_dfs.append(cell)
        death_df = pd.concat(death_dfs)
        # all_dead_df = all_dead.reset_index()
        all_live_df = all_live.reset_index()

        print('all dead', len(all_dead), 'all live', len(all_live), 'zombies', len(zombies), 'deaths',
              len(death_dict))
        print('Time of Death Counts', np.unique(death_lst, return_counts=True))
        death_groups = death_df.groupby(['label', 'Timepoint'])
        x, y = [], []
        # for lbl in range(1, 13):
        #     for i in range(1, 8):
        #         ddf = death_df[(death_df.Timepoint == i) & (death_df.label == lbl)]
        #         y.append(len(ddf))
        #         x.append(i)
        #         plt.plot(x,y, label=lbl)
        # plt.show()
        for i, death_group in death_groups:
            print(f'Column + Timepoint: {i}, Deaths: {len(death_group)}')

        # merge all live with deaths. Ignore all dead and zombie dict

        for name, lst in death_dict.items():
            if name not in zombie_dict:  # filter out zombies
                df = lst[-1]
                time_of_death = lst[0]
                # df['time_of_death'] = time_of_death
                w = df.Sci_WellID.iloc[0]
                # well_int = ascii_uppercase.index(w[0])
                # numeric_well = str(well_int + 1) + w[1:]
                df = pd.DataFrame(
                    ##### switch gedi label such that 0->1 and 1->0, so 1 represents is_dead  #####
                    {'is_dead': [1], 'well': [w], 'label': [df.label.iloc[0]],
                     'condition': [df.condition.iloc[0]],
                     'time_of_death': [time_of_death]})
                # reduce to a row
                regression_dfs.append(df)
        # all_live_df = all_live_df[['Sci_WellID', 'label', 'condition']]

        # all_live_df = all_live_df.rename(columns={'Sci_WellID': 'well'})
        # regression_dfs.append(all_live_df)
        all_live_df['label'] = ''
        all_live_df['condition'] = ''
        # groups
        if use_samples:
            for SampleID, grp in plate_groups:  # todo: redundant
                wells = grp.Sci_WellID
                for w in wells:
                    all_live_df.loc[all_live_df.Sci_WellID == w, 'label'] = SampleID
                    all_live_df.loc[all_live_df.Sci_WellID == w, 'condition'] = self.label_dict[SampleID]
        all_live_df['is_dead'] = 0
        all_live_df['time_of_death'] = max_timepoint
        # # columns
        # else:
        #     for SampleID in range(1, 30):  # todo: redundant
        #         wells = [ascii_uppercase[x] + str(SampleID).zfill(2) for x in range(26)]
        #         for w in wells:
        #             all_live_df.loc[all_live_df.Sci_WellID == w, 'label'] = SampleID

        for i, row in all_live_df.iterrows():
            df = pd.DataFrame(
                #### is not dead, therefore is_dead=0 ####
                {'is_dead': [0], 'well': [row.Sci_WellID], 'label': [row.label], 'condition': [row.condition],
                 'time_of_death': [max_timepoint]})
            regression_dfs.append(df)
        regression_df = pd.concat(regression_dfs)
        return regression_df

    def plot_gedi(self, gedi_cell_data, img_dir, save_dir, well=None, neuron=None):
        data = pd.read_csv(gedi_cell_data, low_memory=False)
        data = data.sort_values(by='Timepoint')
        if well is None:
            w = random.sample(data.Sci_WellID.unique().tolist(), 1)[0]
        else:
            w = well
        data = data[data.Sci_WellID == w]
        # all tracks start at T1
        data = data.groupby(by=['Sci_WellID', 'ObjectLabelsFound']).filter(lambda x: min(x['Timepoint']) == 1)

        # one well, one track, sorted

        groups = data.groupby('Timepoint')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for tp, df in groups:
            # get img
            print(f'Processing Well {w} T{tp}')
            f = glob(os.path.join(img_dir, w, f'*_T{tp}_*.tif'))  # todo: may need to add tag for channel
            assert len(f) < 2, 'multiple files in image directory found by wildcard string matching'
            if len(f):
                f = f[0]
                save_file = os.path.join(save_dir, f.split('/')[-1].split('.')[0] + 'GEDI.png')
                img = imageio.imread(f)
                img = img / np.max(img) * 255
                img = np.uint8(img)
                img = np.dstack((img, img, img))
                for i, row in tqdm(df.iterrows()):
                    # get mask coords
                    x = int(row.BlobCentroidX_RefIntWeighted)
                    y = int(row.BlobCentroidY_RefIntWeighted)
                    r = int(row.Radius)
                    cv2.circle(img, center=(x, y), radius=r, color=(0, 0, 255) if row.gedi else (0, 255, 0),
                               thickness=6)
                if True:
                    imageio.imwrite(save_file, img)
                    print(f'Saved to {save_file}')
                # plt.imshow(img)
                # plt.title(f'Well {w}, Timepoint {tp}, green=dead')
                # plt.show()

                # get live dead


class Survival(trackSurvival):
    def __init__(self, gedicelldata, platemap, savedir):
        super().__init__()
        self.gedicelldata = gedicelldata
        self.platemap = platemap
        self.savedir = savedir

    def regression_df(self, control_group='WTC_iNeuron'):
        # reduce tracks to just one
        regression_df = self.live_dead_tracks(self.gedicelldata, self.platemap)
        print('Running Survival Analysis')
        regression_df = regression_df.drop(columns=['well'])
        regression_df.to_csv(os.path.join(self.savedir, 'survival_data.csv'))
        control_label = None
        for lbl, name in self.label_dict.items():
            if name == control_group:
                control_label = lbl
                break
        assert control_label is not None
        control = regression_df[regression_df.label == control_label]
        data = regression_df[regression_df.label != control_label]
        # results = multivariate_logrank_test(regression_df['time_of_death'], regression_df['label'],
        #                                     regression_df['is_dead'])
        # results.print_summary()
        # group = data.groupby(by='label')
        # cph = CoxPHFitter()
        # cph.fit(regression_df,'time_of_death', event_col='is_dead', robust=True)
        #
        # print('Cox')
        # cph.print_summary()
        # for name, grp in group:
        #     df = pd.concat([control, grp])
        #     cph = CoxPHFitter()
        #     cph.fit(df, 'time_of_death', event_col='is_dead')
        #     print('cox')
        #     cph.print_summary()
        #     print('logrank')
        #     results = multivariate_logrank_test(df['time_of_death'], df['label'],
        #                                         df['is_dead'])
        #     results.print_summary()
        # cph.plot()

        kmf = KaplanMeierFitter()
        ax = plt.subplot(111)
        for lbl, grouped_df in regression_df.groupby('label'):
            kmf.fit(grouped_df["time_of_death"], grouped_df["is_dead"], label=self.label_dict[lbl])
            ax = kmf.plot_survival_function(ax=ax)
        plt.savefig(os.path.join(self.savedir, 'kaplanmeier.png'))
        print('Done')


# todo: check intensity filter
# todo: clean up curation program for cells
# todo: check tracks with curation program
# todo: run a few plates with gedi and this program
# todo: significance on this plate

if __name__ == '__main__':

    result = pyfiglet.figlet_format("Survival", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Deploy GEDICNN model')
    # parser.add_argument('--parent', action="store",
    #                     default='/run/media/jlamstein/data/GEDI-ORDER',
    #                     help='parent directory for Gedi-CNN',
    #                     dest='parent')
    parser.add_argument('--gedicsv', action="store",
                        # default='/run/media/jlamstein/data/GEDI-ORDER/deploy_results/IMG-coculture-2-061522-Th3.csv',
                        default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/ReN_cells_GraceFoundation/GXYTMP/RGEDI_100522/GEDI/deploy_results/deploy.csv',
                        help='path to gedi cnn results', dest="gedicsv")
    parser.add_argument('--platemap', action="store",
                        default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/platelayout_iMGcoculture.csv',
                        help='path to platemap', dest="platemap")
    parser.add_argument('--img_dir', action="store",
                        # default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/iMG-coculture-1-061522/AlignedImages',
                        default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3/AlignedImages',
                        help='path to image directory, aligned images most likely', dest="img_dir")
    parser.add_argument('--save_dir', action="store",
                        default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3/GEDIImages',
                        help='path to image directory, aligned images most likely', dest="save_dir")
    parser.add_argument('--trackcsv', action="store",
                        # default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3/cell_data.csv',
                        default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/ReN_cells_GraceFoundation/GXYTMP/RGEDI_100522/OverlaysTablesResults/cell_data.csv',
                        help='path to cell data with tracking info.', dest="trackcsv")
    parser.add_argument('--gedi_cell_data', action="store",
                        default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3/gedi_cell_data.csv',
                        help='path to cell data with tracking info.', dest="gedi_cell_data")
    parser.add_argument('--resdir', action="store",
                        default='/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3',
                        help='results directory', dest="resdir")
    parser.add_argument('--merge_csvs', type=int, action="store", default=True,
                        help='Merge gedicsv with trackcsv in cell data. Saved to gedicelldata.',
                        dest="merge_csvs")
    parser.add_argument('--plot_bool', type=int, action="store", default=False,
                        help='Get plots with gedi predictions projected on montaged image',
                        dest="plot_bool")
    parser.add_argument('--survival_bool', type=int, action="store", default=True,
                        help='Get survival data and plots',
                        dest="survival_bool")

    args = parser.parse_args()
    print('ARGS:\n', args)

    # trackcsv = '/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/iMG-coculture-1-061522/OverlaysTablesResults/cell_data.csv'
    # gedi_cell_data = '/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/iMG-coculture-1-061522/OverlaysTablesResults/gedi_cell_data.csv'
    TS = trackSurvival()
    if args.merge_csvs:
        TS.merge_dfs(args.gedicsv, args.trackcsv, args.gedi_cell_data)
    # if args.survival_bool:
    #     TS.live_dead_tracks(args.gedicelldata, args.platemap)
    if args.plot_bool:
        TS.plot_gedi(args.gedi_cell_data, args.img_dir, args.save_dir)
    if args.survival_bool:
        Sur = Survival(args.gedi_cell_data, args.platemap, args.resdir)
        Sur.regression_df()
    # todo: survival curves for different conditions with plot and csv raw data
