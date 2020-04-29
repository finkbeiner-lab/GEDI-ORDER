Gedi Cnn in TF 2.0
CNN is a binary classifier that uses cell morphology to determine death. 
Model may be trained, retrained from a base model, and deployed. 

The original data, already preprocessed into tfrecords, is available on the NAS as is the origianl gedi model in 
/mnt/finkbeinerlab/robodata/GEDI_CLUSTER.

To run, create a virtual environment with pip. This repo was created with python 3.6. Pip is currently preferred 
over anaconda because pip has support for the package tf-addons, but conda does not. The tensorflow version is 2.0, with cuda version 10.1. 
