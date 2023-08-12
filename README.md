# Music Note Classification with CNN

This repository contains a Convolutional Neural Network (CNN) model for classifying different musical notes. The CNN model is trained on a dataset of music symbols, enabling automated recognition of note types in sheet music.

Integratring with a simple GUI, this software allows you to simply classify different notes and get a short tutorial on each type of note.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Project Overview

In the realm of music notation, the automated recognition of musical notes is a fascinating application of deep learning. This project aims to classify different music symbols representing notes using a CNN model. The trained model can provide insights into the types of notes present in sheet music.

## Dataset

The dataset used for training and validation comes from the [Music Notes Datasets](https://www.kaggle.com/datasets/kishanj/music-notes-datasets?resource=download) available on Kaggle. It contains five different music symbols:

1. Whole Note
2. Half Note
3. Quarter Note
4. Eight Note
5. Sixteenth Note

Each category consists of 1000 data samples, with 800 for training and 200 for validation.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/rambodazimi/music-note-classification.git

2. Navigate to the project directory:
   ```bash
   cd music-note-classification
   
## Usage

The GUI application allows you to upload an image containing a musical note, predicts its class, and provides a description of the note.
Just run the Python file using the following command on your terminal:
   ```bash
   python model.py
   ```

## Model Architecture

The CNN model architecture used for music note classification is as follows:

Layer (type)               Output Shape         Param

conv2d (Conv2D)            (None, 62, 62, 32)   896

...

dense_1 (Dense)            (None, 5)            645

Total params: 1,240,869

Trainable params: 1,240,869

Non-trainable params: 0

## Results

After training for 10 epochs, the model achieved a validation accuracy of approximately 0.9, demonstrating its effectiveness in classifying musical notes.



