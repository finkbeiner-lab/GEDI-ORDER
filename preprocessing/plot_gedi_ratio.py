"""Read csv and plot gedi ratio"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

f = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/H23-10uM_GEDI.csv'
f = '/mnt/linsley/Shijie/Galaxy-temp-NSCLC_ML/GXYTMP-2022-0429-NSCLC-H23-10uM/H23-10uM_GEDI.csv'
f = '/mnt/linsley/Soothsayer/GEDI_RATIO/1703-10uM_GEDI.csv'
f = '/mnt/linsley/Soothsayer/GEDI_RATIO/NSCLC-half-H23-10uM_GEDI.csv'

df = pd.read_csv(f)
print(df.columns)
print(pd.unique(df.GEDILabel))
df.Timepoint = df.Timepoint + np.random.random((len(df.Timepoint)))/2
df = df[df.GEDI_confocal < 5]
df.plot.scatter(x='Timepoint', y='GEDI_confocal', s=1, c='DarkBlue')
plt.show()