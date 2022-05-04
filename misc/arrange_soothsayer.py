"""Move or delete this file
Combine tracking csvs with gedi ratio csvs
Group cells into time
Rename crops based on voronoi tracking
Move crops to live/dead folder
Train"""

import pandas as pd


# Combine tracking csvs with gedi ratio csvs
# pd merge
class MergeTrackingWithGedi:
    def __init__(self, tracking_csv, gedi_csv, gedi_csv2, condition=None):
        self.track = pd.read_csv(tracking_csv)
        self.track = self.track.loc[:, ~self.track.columns.str.contains('^Unnamed')]
        self.gedi = pd.read_csv(gedi_csv)
        self.gedi = self.gedi.loc[:, ~self.gedi.columns.str.contains('^Unnamed')]
        self.gedi = self.gedi[['Sci_WellID', 'Sci_PlateID', 'ObjectLabelsFound', 'MeasurementTag', 'Timepoint',
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
            self.gedi2 = self.gedi2[['Sci_WellID', 'Sci_PlateID', 'ObjectLabelsFound', 'MeasurementTag', 'Timepoint',
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

#Group cells into time
#Rename crops based on voronoi tracking

class RenameCropsWithTrackID:
    def __init__(self, f):
        self.df = pd.read_csv(f)
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]

        #



if __name__ == '__main__':
    tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/Voronoi/2022_05_03_17_51_02_cell_data_TRACKED.csv'
    gedi_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/H23-10uM_GEDI.csv'
    gedi_csv2 = None
    condition = ['H23']

    # tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-1703-10uM/Voronoi/2022_05_03_18_10_39_cell_data_TRACKED.csv'
    # gedi_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-1703-10uM/1703-10uM_GEDI.csv'
    # gedi_csv2 = None
    # condition = ['1703']

    # tracking_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM/Voronoi/2022_05_04_10_36_31_cell_data_TRACKED.csv'
    # gedi_csv = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM/NSCLC-half-1703-10uM_GEDI.csv'
    # gedi_csv2 = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-10uM/NSCLC-half-H23-10uM_GEDI.csv'
    # condition = ['1703', 'H23']

    # MRG = MergeTrackingWithGedi(tracking_csv, gedi_csv, gedi_csv2, condition)
    # MRG.run()
