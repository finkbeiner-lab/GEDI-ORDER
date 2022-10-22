"""Process curation from gedi_crops_to_curation

Fiji curator in /gladstone/finkbeiner/linsley/CURATION_FIJI, also on dropbox"""

import pandas as pd
from glob import glob
import os
from tqdm import tqdm


class Review:
    def __init__(self, gedicsvs, csv_log, savepath):
        self.gedicsvs = gedicsvs
        self.csv_log = csv_log
        self.savepath = savepath

    def check(self):
        dfs = []
        for csv in self.gedicsvs:
            dfs.append(pd.read_csv(csv))
        df = pd.concat(dfs, ignore_index=True)

        tokens = df.filepath.str.split('/', expand=True)
        df['filename'] = tokens[len(tokens.columns) - 1]
        df.sort_values('filename',inplace=True)
        csvs = glob(os.path.join(self.csv_log, '*.csv'))
        curations = []
        for csv in csvs:
            curations.append(pd.read_csv(csv))
        curation = pd.concat(curations, ignore_index=True)
        curation[['batch', 'filename']] = curation.Fname.str.split('/', expand=True)
        curation['filepath'] = ''
        curation['prediction'] = -1
        curation['agree'] = -1
        cnt = 0
        curation.sort_values('filename', inplace=True)
        validation = curation.copy()
        for j, row in curation.iterrows():
            d = df[row.filename==df.filename]
            idx = d.index[0]
            validation.loc[j,'filepath'] = df.loc[idx, 'filepath']
            validation.loc[j, 'prediction'] = df.loc[idx, 'prediction']
            validation.loc[j,'agree'] = row.Curation % 2==df.loc[idx, 'prediction'] # pred: 0:dead, 1: live, curation: 1:live, 2:dead
        validation.to_csv(self.savepath, index=False)

        print(f'Saved to {self.savepath}')


if __name__ == '__main__':
    gedicsvs = ['/gladstone/finkbeiner/linsley/GEDI_CLUSTER/deploy_results/IMG-coculture-2-061522-Th3.csv',
                '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/deploy_results/iMG-coculture-1-061522.csv']
    csv_log = '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/Curation/Josh/csv_logs'
    savepath = '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/Curation/Results'
    savepath = os.path.join(savepath, csv_log.split('/')[-2] + '.csv')
    Rev = Review(gedicsvs, csv_log, savepath)
    Rev.check()
