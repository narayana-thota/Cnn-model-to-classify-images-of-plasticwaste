# CNN Model for Image Classification  

This project involves building a Convolutional Neural Network (CNN) for image classification. The primary goal is to classify images into predefined categories with high accuracy. This README outlines the work done during the week, setup instructions, and relevant details.  

## Table of Contents  

1. [Overview](#overview)  
2. [This Week's Work](#this-weeks-work)  
3. [Technologies Used](#technologies-used)
4.  [Dataset Link](#dataset-link)
5. [Setup Instructions](#setup-instructions)  
6. [Contributors](#contributors)  

## Overview  

The CNN model is designed to process image data and extract meaningful features for classification. It uses layers such as convolution, pooling, and dense layers to learn patterns from the dataset.  

## This Week's Work  

During this week, the following tasks were completed:  
1. **Model Architecture:**  
   - Designed the CNN architecture with layers including:
     - Convolutional layers  
     - Max Pooling layers  
     - Fully Connected layers  
   - Included `ReLU` activation for non-linearity.  
2. **Dataset Preparation:**  
   - Loaded and preprocessed the dataset (resizing, normalization).  
3. **Model Training:**  
   - Trained the model on the prepared dataset with validation split.  
   - Used an optimizer (`Adam`) and calculated loss (`categorical_crossentropy`).  
4. **Performance Metrics:**  
   - Evaluated the model using accuracy and loss on the test set.  

## Technologies Used  

- **Programming Language:** Python  
- **Libraries/Frameworks:**  
  - TensorFlow/Keras  
  - NumPy  
  - Matplotlib  
  - Pandas (if applicable for data handling)
## Dataset Link
   [Dataset Link](https://www.kaggle.com/datasets/techsash/waste-classification-data/data)
## Setup Instructions  

1. Clone the repository:  
   ```bash  
   https://github.com/narayana-thota/Cnn-model-to-classify-images-of-plasticwaste.git
2. Navigate to the project directory:
   ```bash
   cd cnn-model  
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt  
4. Run the model training script:
   ```bash
   python train.py  
5. Evaluate the model on the test set:
   ```bash
   python evaluate.py  
## Contibutors
   Thota Om Sada Siva Venkata Narayana

