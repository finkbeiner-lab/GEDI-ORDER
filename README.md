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

## Run GEDI-CNN on Your Data

### Get the Code
Go to the [GEDI-ORDER github](https://github.com/finkbeiner-lab/GEDI-ORDER) and clone the repo. 

### Dependencies
The code is tested in python 3.6 using pip to install packages. Instructions for installing pip may be found [here](https://pip.pypa.io/en/stable/installing/).

The code uses dependencies found in requirements.txt. You may install the dependencies with:
    pip install -r requirements.txt
Depending on your computer system, you may need to replace `pip` with `pip3`, and Windows, Mac and Linux may have different names for modules, such as matplotlib may be referred to as python-matplotlib.

If you're installing without requirements.txt, the notable dependencies are:
1. tensorflow-gpu==2.0.0 (if you don't use a gpu for machine learning, install tensorflow==2.0.0 instead)
2. imageio==2.8.0
3. opencv-python==4.2.0.34
4. pandas==1.0.3
5. matplotlib==3.2.1

If you're using CUDA with a gpu, tensorflow-gpu 2.0 pairs with CUDA 10.0. 

### Run the Code 
_We are currently working to make GEDI-CNN more user-friendly. Stay tuned._

