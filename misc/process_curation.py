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
        self.curation_path = savepath

    def check(self):
        dfs = []
        for csv in self.gedicsvs:
            dfs.append(pd.read_csv(csv))
        df = pd.concat(dfs, ignore_index=True)

        tokens = df.filepath.str.split('/', expand=True)
        df['filename'] = tokens[len(tokens.columns) - 1]
        df.sort_values('filename', inplace=True)
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
            d = df[row.filename == df.filename]
            idx = d.index[0]
            validation.loc[j, 'filepath'] = df.loc[idx, 'filepath']
            validation.loc[j, 'prediction'] = df.loc[idx, 'prediction']
            validation.loc[j, 'agree'] = row.Curation % 2 == df.loc[
                idx, 'prediction']  # pred: 0:dead, 1: live, curation: 1:live, 2:dead
        validation.to_csv(self.curation_path, index=False)

        print(f'Saved to {self.curation_path}')

    def compare(self, curation_dir):
        csvs = glob(os.path.join(curation_dir, '*.csv'))  # get all csvs
        dfs = {}
        df_lst = []
        keep_cols = ['Curation', 'Fname', 'batch', 'filename', 'filepath', 'prediction']
        for csv in csvs:
            df = pd.read_csv(csv)
            delete_columns = [c for c in df.columns if c not in keep_cols]
            df = df.drop(columns=delete_columns)
            dfs[csv.split('/')[-1].split('.')[0]] = df
        # cols = ['Curation', 'Fname', 'Idx', 'batch', 'filename', 'filepath', 'prediction']
        for name, df in dfs.items():
            df = df.rename(columns={'Curation': f'Curation_{name}'})
            df_lst.append(df)
        # df_lst = [df.set_index('filepath') for df in df_lst]

        df = df_lst[0]
        for _df in df_lst[1:]:
            df = pd.merge(df, _df, on=['Fname', 'filename', 'filepath', 'prediction', 'batch'], how='inner')
        # df_lst[0].join(df_lst[1:], how='inner')
        print(df)
        curation_cols = [c for c in df.columns if 'Curation' in c]
        BG = df[df.Curation_Bianca == df.Curation_Gracie]
        not_BG = df[df.Curation_Bianca != df.Curation_Gracie]
        BJ = df[df.Curation_Bianca == df.Curation_Josh]
        not_BJ = df[df.Curation_Bianca != df.Curation_Josh]
        BGJ = df[(df.Curation_Bianca == df.Curation_Gracie) & (df.Curation_Bianca == df.Curation_Josh)]
        print(f'Bianca and Gracie agree: {len(BG)}')
        print(f'Bianca and Gracie disagree: {len(not_BG)}')
        print(f'Bianca and Josh agree: {len(BJ)}')
        print(f'Bianca and Josh disagree: {len(not_BJ)}')
        print(f'Bianca Gracie Josh agree: {len(BGJ)}')

        print(f'Bianca and Gracie agree Live: {len(BG[BG.Curation_Bianca == 1])}')
        print(f'Bianca and Gracie agree Dead: {len(BG[BG.Curation_Bianca == 2])}')

        curation_cols = ['Curation_Bianca', 'Curation_Josh', 'Curation_Gracie']

        agree = {c: [] for c in curation_cols}
        pos_agree = {c: [] for c in curation_cols}
        neg_agree = {c: [] for c in curation_cols}
        acc = {c: [] for c in curation_cols}
        pos_acc = {c: [] for c in curation_cols}
        neg_acc = {c: [] for c in curation_cols}
        agree['curation'] = curation_cols
        pos_agree['curation'] = curation_cols
        neg_agree['curation'] = curation_cols
        acc['curation'] = curation_cols
        pos_acc['curation'] = curation_cols
        neg_acc['curation'] = curation_cols
        for c in curation_cols:
            for c2 in curation_cols:
                if c == c2:
                    agree[c].append(len(df))
                    pos_agree[c].append(len(df[df.prediction == 1]))
                    neg_agree[c].append(len(df[df.prediction == 0]))
                    acc[c].append(len(df[df[c] % 2 == df.prediction]))
                    pos_acc[c].append(len(df[(df[c] % 2 == df.prediction) & (df.prediction == 1)]))
                    neg_acc[c].append(len(df[(df[c] % 2 == df.prediction) & (df.prediction == 0)]))
                else:
                    agree[c].append(len(df[df[c] == df[c2]]))
                    pos_agree[c].append(len(df[(df[c] == df[c2]) & (df[c] == 1)]))
                    neg_agree[c].append(len(df[(df[c] == df[c2]) & (df[c] == 2)]))
                    acc[c].append(len(df[(df[c] == df[c2]) & (df[c] % 2 == df.prediction)]))
                    pos_acc[c].append(len(df[(df[c] == df[c2]) & (df.prediction == 1)]))
                    neg_acc[c].append(len(df[(df[c] == df[c2]) & (df.prediction == 0)]))
        pd.DataFrame(agree).set_index('curation').to_csv(os.path.join(curation_dir, 'Analysis', 'agree.csv'))
        pd.DataFrame(pos_agree).set_index('curation').to_csv(os.path.join(curation_dir, 'Analysis', 'pos_agree.csv'))
        pd.DataFrame(neg_agree).set_index('curation').to_csv(os.path.join(curation_dir, 'Analysis', 'neg_agree.csv'))
        pd.DataFrame(acc).set_index('curation').to_csv(os.path.join(curation_dir, 'Analysis', 'acc.csv'))
        pd.DataFrame(pos_acc).set_index('curation').to_csv(os.path.join(curation_dir, 'Analysis', 'pos_acc.csv'))
        pd.DataFrame(neg_acc).set_index('curation').to_csv(os.path.join(curation_dir, 'Analysis', 'neg_acc.csv'))


if __name__ == '__main__':
    gedicsvs = ['/gladstone/finkbeiner/linsley/GEDI_CLUSTER/deploy_results/IMG-coculture-2-061522-Th3.csv',
                '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/deploy_results/iMG-coculture-1-061522.csv']
    csv_log = '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/Curation/Gracie/csv_logs'
    curation_dir = '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/Curation/Results'
    curation_path = os.path.join(curation_dir, csv_log.split('/')[-2] + '.csv')
    Rev = Review(gedicsvs, csv_log, curation_path)
    # Rev.check()
    Rev.compare(curation_dir)
