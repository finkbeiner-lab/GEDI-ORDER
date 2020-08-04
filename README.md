# GEDI-CNN: Genetically Encoded Death Indicator Convolutional Neural Network

## About

### Convolutional Neural Network

The Convolutional Neural Network (CNN) is a binary classifier that predicts if a neuron is alive or dead. The CNN relies on cell morphology to judge a cell's vitality and uses an image of a cell to run. Life and death for a cell can be difficult to define; the GEDI biomarker, described below, is used as the ground truth. With a curator, thousands of images may be generated with the GEDI biomarker, which we argue is more reliable than human curation. 

### GEDI: Genetically Encoded Death Indicator

Cell death is a critical process that occurs normally in health and disease. GEDI specifically detects an intracellular Ca2+ level that cells achieve early in the cell death process and marks a stage at which cells are irreversibly committed to die. The time-resolved nature of GEDI delineates a binary demarcation of cell life and death in real time, reformulating the definition of cell death. We demonstrate that GEDI acutely and accurately reports death of rodent and human neurons in vitro, and show GEDI enables a novel automated imaging platform for single cell detection of neuronal death in vivo in zebrafish larvae. With a quantitative pseudo-ratiometric signal, GEDI facilitates high-throughput analysis of cell death in time lapse imaging analysis, providing the necessary resolution and scale to identify early factors leading to cell death in studies of neurodegeneration.

The paper is available at https://www.biorxiv.org/content/10.1101/726588v1.

Gedi Cnn in TF 2.0
CNN is a binary classifier that uses cell morphology to determine death. 
Model may be trained, retrained from a base model, and deployed. 

The original data, already preprocessed into tfrecords, is available on the NAS as is the origianl gedi model in 
/mnt/finkbeinerlab/robodata/GEDI_CLUSTER.

To run, create a virtual environment with pip. This repo was created with python 3.6. Pip is currently preferred 
over anaconda because pip has support for the package tf-addons, but conda does not. The tensorflow version is 2.0, with cuda version 10.1. 
