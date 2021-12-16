# GEDI-CNN: Genetically Encoded Death Indicator Convolutional Neural Network

## What the GEDI-CNN Does

The Convolutional Neural Network (CNN) is a binary classifier that predicts if a neuron is alive or dead. The CNN relies on cell morphology to judge a cell's vitality and uses an image of a cell to run. Life and death for a cell can be difficult to define; the GEDI biomarker, described below, is used as the ground truth. With a curator, thousands of images may be generated with the GEDI biomarker, which we argue is more reliable than human curation. 

![Neuron](/examples/neuron.png)

The image are two examples of neurons predicted to be live. 

## Video: Biomarker Optimized Convolutional Neural Networks

Presented by Jeremy Linsley.
[![Biomarker Optimized Convolutional Neural Networks](/examples/bocnn_video.PNG)](https://youtu.be/v_vf1eisGr4 "Biomarker Optimized Convolutional Neural Networks")

## How the Genetically Encoded Death Indicator (GEDI) Works

Cell death is a critical process that occurs normally in health and disease. GEDI specifically detects an intracellular Ca2+ level that cells achieve early in the cell death process and marks a stage at which cells are irreversibly committed to die. The time-resolved nature of GEDI delineates a binary demarcation of cell life and death in real time, reformulating the definition of cell death. 

The paper, Genetically encoded cell-death indicators (GEDI) to detect an early irreversible commitment to neurodegeneration, is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/726588v1).

## How It Works

Neural networks can be frustrating because they are essentially a black box. The millions of weights multiplied in linear algebra can appear unrelated to biological functions. This work includes GRAD-CAM with guided backpropagation to visualize what of the neuron the neural network weights with importance. This visualization can give new and powerful insights into biological functions if the computer is seeing subtle something that the human eye does not. 

In the image below, green is neuron inputted into the GEDI-CNN, blue is weighted as a death signal, and red is a live signal.  

![Gradcam](/examples/gradcam.png)



### Get the Code
Go to the [GEDI-ORDER github](https://github.com/finkbeiner-lab/GEDI-ORDER) and clone the repo. 

## Next, download the following model weight files from the below links
The dropbox link is: 
https://www.dropbox.com/s/4xzfu59cc48lf8y/gedicnn.h5?dl=0
You will find the model file, gedicnn.h5.

### Dependencies
The code is tested in python 3.6 using pip to install packages. Instructions for installing pip may be found [here](https://pip.pypa.io/en/stable/installing/).

The code uses dependencies found in requirements.txt. You may install the dependencies with:
    pip install -r requirements.txt
Depending on your computer system, you may need to replace `pip` with `pip3`, and Windows, Mac and Linux may have different names for modules, such as matplotlib may be referred to as python-matplotlib.

If you're installing without requirements.txt, the notable dependencies are:
1. tensorflow-gpu
2. imageio
3. opencv-python
4. pandas
5. matplotlib

The code has been tested in python 3.6 and 3.7 with tensorflow-gpu 2.0, 2.3, and 2.5. 

## Run GEDI-CNN on Your Data
To run the GEDI-CNN, run deploy.py.

In param.py, set paths for your machine. Ensure that base_gedi pointo your path to the gedicnn.h5 downloaded from dropbox. 

The deploy file takes arguments:
python ../deploy/deploy.py --parent PARENT_DIRECTORY \
--im_dir IMAGE_DIRECTORY \
--model_path PATH_TO_gedicnn.h5_downloaded_from_dropbox \
--resdir DIRECTOR_TO_STORE_RESULTS \
--preprocess_tfrecs BOOL_TO_GENERATE_TFRECORDS \
--use_gedi_cnn BOOL_TRUE_IF_USING_GEDICNN

The deploy script takes a single image directory to run on. The script converts the images into a tfrecord. If that tfrecord has not been generated, 
set preprocess_tfrecs to True, otherwise, you need not create a tfrecord as it already exists, and sett preprocess_tfrecs to False. 


## Finetuning
If you want to retrain the GEDICNN, run the train file. 

python ../main/train.py --datadir PARENT_DIRECTORY \
--pos_dir LIST_OF_DIRECTORIES_WITH_IMAGE_CLASS_1 \
--neg_dir LIST_OF_DIRECTORIES_WITH_IMAGE_CLASS_0 \
--balance_method OPTIONAL_METHOD_TO_CUTOFF_OR_MULTIPY_SAMPLES_IN_UNBALANCED_DATA \
--use_neptune BOOL_TO_USE_NEPTUNE_AI_FOR_LOGGING \
--retrain TRUE_TO_REUSE_GEDI_WEIGHTS_FOR_FINETUNING

## Run Gradcam

The saliency map or activation map may be seen with gradcam with guided backpropagation. The gradcam script is written in 
tensorflow 1.x and uses compat mode with tensorflow 2.x. Gradcam takes either a tfrecord or image directory. 

Run gradcam: 

python ../activationmap/gradcam.py --im_dir IMAGE_DIRECTORY_TO_RUN_LEAVE_BLANK_IF_USING_TFRECORD \
--model_path SET_PATH_TO_MODEL \
--deploy_tfrec PATH_TO_TFRECORD_IF_USING_IMAGE_DIR_LEAVE_BLANK \
--layer_name LAYER_NAME_OF_MODEL_TO_VISUALIZE_DEFAULT_IS_block5_conv3_FOR_VGG16 \
--resdir DIRECTORY_FOR_YOUR_RESULTS \
--imtype SUFFIX_FOR_IMAGE_TYPE (tif, jpg, png)

If using the gedicnn, set the model path to the gedicnn.h5. 


### Citing
To cite please use [![DOI](https://zenodo.org/badge/253638158.svg)](https://zenodo.org/badge/latestdoi/253638158)


