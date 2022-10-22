"""Add gedi crops to fiji curator

Fiji curator in /gladstone/finkbeiner/linsley/CURATION_FIJI, also on dropbox"""
import os
import glob
import pandas as pd
from typing import List
import shutil


class Curator:
    def __init__(self, gedicsvs: List, savedir: str):
        self.csvs = gedicsvs
        self.cnt = 0
        self.batch = None
        self.batch_div = 100
        assert 'Images' in savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.savedir = savedir

    def get_data(self, N: int):
        """Loads gedicsvs. Randomly samples 50/50 live / dead. Copies crops to save directory.
        N (int): sample N per class"""

        # load csvs
        dfs = []
        for csv in self.csvs:
            dfs.append(pd.read_csv(csv))

        for df in dfs:
            m = min(df.groupby('prediction').size())
            if m < N: N = m
            neg = df[df.prediction == 0].sample(n=N, random_state=121)
            pos = df[df.prediction == 1].sample(n=N, random_state=121)
            data = pd.concat([neg, pos])
            for i, row in data.iterrows():
                if self.cnt % self.batch_div == 0:
                    self.batch = 'batch_' + str(1 + self.cnt // self.batch_div)
                if not os.path.exists(os.path.join(self.savedir, self.batch)):
                    os.mkdir(os.path.join(self.savedir, self.batch))
                f = row.filepath
                dst = os.path.join(self.savedir, self.batch, f.split('/')[-1])
                shutil.copyfile(f, dst)
                self.cnt += 1

        print(f'Crops for curation sent to: {self.savedir}')


if __name__ == '__main__':
    gedicsvs = ['/gladstone/finkbeiner/linsley/GEDI_CLUSTER/deploy_results/IMG-coculture-2-061522-Th3.csv',
                '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/deploy_results/iMG-coculture-1-061522.csv']
    savedir = '/gladstone/finkbeiner/linsley/GEDI_CLUSTER/Curation/Josh/Images'
    Cur = Curator(gedicsvs, savedir)
    Cur.get_data(N=100)
