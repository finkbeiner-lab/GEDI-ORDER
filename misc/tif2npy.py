import imageio
import numpy
import os
import glob

guided = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit/10_single/live_false/PID20160831_RGEDIMachineLearning3_T1_24-0_B11_0_FITC-DFTrCy5_BGs_MN_ALIGNED_4.tif'
raw = '/mnt/finkbeinerlab/robodata/Gennadi/batches16bit/10/10_single/PID20160831_RGEDIMachineLearning3_T1_24-0_B11_0_FITC-DFTrCy5_BGs_MN_ALIGNED_4.tif'
notguided = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit_notguided/10_single/live_false/PID20160831_RGEDIMachineLearning3_T1_24-0_B11_0_FITC-DFTrCy5_BGs_MN_ALIGNED_4.tif'

g = imageio.imread(guided)
r = imageio.imread(raw)
n = imageio.imread(notguided)
parent = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit'
numpy.save( os.path.join(parent, 'guided.npy'), g)
numpy.save(os.path.join(parent, 'original.npy'), r)
numpy.save(os.path.join(parent, 'notguided.npy'), n)
