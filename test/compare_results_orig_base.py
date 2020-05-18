"""
Compare the results of the original gedi model and the 2.0 base model (h5 format).
"""

import pandas as pd

# '/mnt/finkbeinerlab/robodata/GalaxyTEMP/BSMachineLearning_TestCuration/batches/1'
orig = '/mnt/data/GEDI_RESULTS/project_files/results/trained_gedi_model/5.csv'
new = '/mnt/data/GEDI-ORDER/deploy_results/BSMachineLearning_TestCuration_5.csv'

o = pd.read_csv(orig)
n = pd.read_csv(new)
miss = 0
correct = 0
for i, row in o.iterrows():
    f = row.files
    o_pred = row.live_guesses
    n_pred = n.loc[n.filepath.str.contains(f), 'prediction'].to_numpy()
    try:
        n_pred = n_pred[0]
    except IndexError:
        print('err')
        print(f)
    if  o_pred!=n_pred:
        miss += 1
        print(f)
        # print('1.x', o_pred)
        # print('2.0', n_pred)
    else:
        correct += 1
print('total missed: {}'.format(miss))
print('total correct: {}'.format(correct))