import os
from glob import glob

cls = ['one_true', 'one_false', 'zero_true', 'zero_false']
cnt_dict = {'one_true':{'H23':0, '1703':0}, 'one_false':{'H23':0, '1703':0}, 'zero_true':{'H23':0, '1703':0}, 'zero_false':{'H23':0, '1703':0}}
for c in cls:
    folder = f'/mnt/linsley/Soothsayer/Gradcam/Sooth/{c}'
    files = glob(os.path.join(folder, '*_2022-0429-NSCLC-10uM_*.tif'))
    for f in files:
        well = f.split('/')[-1].split('_')[4]
        col = int(well[1:])
        if col < 7:
            cnt_dict[c]['H23']+= 1
        else:
            cnt_dict[c]['1703']+= 1

print(cnt_dict)