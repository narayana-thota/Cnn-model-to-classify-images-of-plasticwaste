Waste Management Using CNN Model
This project involves building a Convolutional Neural Network (CNN) model for waste management. The goal is to classify images of waste items into two categories: Organic and Recyclable. This README outlines the work done during the week, setup instructions, and relevant details.

Table of Contents
Overview
This Week's Work
Technologies Used
Setup Instructions
Dataset Link
Contributors
Overview
The CNN model is designed to process images of waste items and classify them into two categories:

Organic
Recyclable
The model leverages deep learning techniques to extract features from images and use those features to perform image classification. This approach aims to automate the waste sorting process, making it more efficient and accurate.

This Week's Work
During this week, the following tasks were completed:

Data Preparation:

Collected and preprocessed the dataset from the TRAIN and TEST directories.
Images were resized and normalized before being fed into the model.
Data augmentation using the ImageDataGenerator was applied for better model generalization.
Model Architecture:

Designed a CNN architecture with the following layers:
Convolutional layers with ReLU activation
Max pooling layers for downsampling
Fully connected layers for classification
Dropout layers for regularization
The final output layer used a sigmoid activation to classify into two categories: Organic or Recyclable.
Model Training:

Trained the model for 1 epoch using the Adam optimizer and binary cross-entropy loss function.
Used batch size of 256 for training.
Visualization:

Displayed a pie chart showing the distribution of labels in the dataset (Organic vs. Recyclable).
Plotted random sample images from the dataset to visually check their labels.
Performance Metrics:

Evaluated the model on the validation set to track accuracy and loss.
Technologies Used
Programming Language: Python
Libraries/Frameworks:
TensorFlow/Keras
OpenCV
NumPy
Pandas
Matplotlib
tqdm (for progress bars)
Dataset Link
The dataset used for this project is available at the following link:
Dataset Link
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
## Contributors
Thota Om Sada Siva Venkata Narayana
   
   
