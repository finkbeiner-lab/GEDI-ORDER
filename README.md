# GEDI-CNN: Genetically Encoded Death Indicator Convolutional Neural Network

## About

### What the GEDI-CNN Does

The Convolutional Neural Network (CNN) is a binary classifier that predicts if a neuron is alive or dead. The CNN relies on cell morphology to judge a cell's vitality and uses an image of a cell to run. Life and death for a cell can be difficult to define; the GEDI biomarker, described below, is used as the ground truth. With a curator, thousands of images may be generated with the GEDI biomarker, which we argue is more reliable than human curation. 

### How the Genetically Encoded Death Indicator (GEDI) Works

Cell death is a critical process that occurs normally in health and disease. GEDI specifically detects an intracellular Ca2+ level that cells achieve early in the cell death process and marks a stage at which cells are irreversibly committed to die. The time-resolved nature of GEDI delineates a binary demarcation of cell life and death in real time, reformulating the definition of cell death. 

The paper, Genetically encoded cell-death indicators (GEDI) to detect an early irreversible commitment to neurodegeneration, is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/726588v1).

## How to Use

### Run GEDI-CNN on Your Data

#### Get the Code
Go to the [GEDI-ORDER github](https://github.com/finkbeiner-lab/GEDI-ORDER) and clone the repo. 

#### Dependencies
The code is tested in python 3.6 using pip to install packages. Instructions for installing pip may be found [here](https://pip.pypa.io/en/stable/installing/).

The code uses dependencies found in requirements.txt. If you're installing without requirements.txt, the notable dependencies are:
1. tensorflow-gpu==2.0.0 (if you don't use a gpu for machine learning, install tensorflow==2.0.0 instead)
2. imageio==2.8.0
3. opencv-python==4.2.0.34
4. pandas==1.0.3
5. matplotlib==3.2.1

#### Run the Code



### 

Gedi Cnn in TF 2.0
CNN is a binary classifier that uses cell morphology to determine death. 
Model may be trained, retrained from a base model, and deployed. 

The original data, already preprocessed into tfrecords, is available on the NAS as is the origianl gedi model in 
/mnt/finkbeinerlab/robodata/GEDI_CLUSTER.

To run, create a virtual environment with pip. This repo was created with python 3.6. Pip is currently preferred 
over anaconda because pip has support for the package tf-addons, but conda does not. The tensorflow version is 2.0, with cuda version 10.1. 
