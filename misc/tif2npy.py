import imageio
import numpy
import os
import glob

guided = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit/guided.tif'
raw = '/mnt/finkbeinerlab/robodata/Gennadi/batches16bit/10/10_single/PID20160831_RGEDIMachineLearning3_T1_24-0_B11_0_FITC-DFTrCy5_BGs_MN_ALIGNED_4.tif'
gradcam = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit/gradcam.tif'
backprop_grads = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit/backprop_grads.tif'

g = imageio.imread(guided)
r = imageio.imread(raw)
cam = imageio.imread(gradcam)
b = imageio.imread(backprop_grads)
parent = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit'
numpy.save( os.path.join(parent, 'guided.npy'), g)
numpy.save(os.path.join(parent, 'original.npy'), r)
numpy.save(os.path.join(parent, 'gradcam.npy'), cam)
numpy.save(os.path.join(parent, 'backprop_grads.npy'), b)
