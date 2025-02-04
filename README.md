# ♻️ Waste Management Using CNN Model 🧠

## 📌 Overview
This project involves building a Convolutional Neural Network (CNN) model for waste management. The goal is to classify images of waste items into different categories, such as **Organic, Recyclable, and various types of plastic waste**. The model leverages deep learning techniques to extract features from images and use those features for accurate classification. This approach aims to **automate the waste sorting process**, making it more efficient and precise. 🚀

---

## 📖 Table of Contents
- 📌 [Overview](#-overview)
- 📅 [This Week's Work](#-this-weeks-work)
- 🛠 [Technologies Used](#-technologies-used)
- ⚙️ [Setup Instructions](#-setup-instructions)
- 📂 [Dataset Link](#-dataset)
- 👥 [Contributors](#-contributors)

---

## 📅 This Week's Work

During this week, the following tasks were completed:

### 📊 Data Preparation:
- 📥 Collected and preprocessed the dataset from the `TRAIN` and `TEST` directories.
- 📏 Images were resized and normalized before being fed into the model.
- 🔄 Data augmentation using **ImageDataGenerator** was applied for better model generalization.

### 🏗 Model Architecture:
The CNN model was designed with the following layers:
- 🧩 **Convolutional layers** with ReLU activation
- 📉 **Max pooling layers** for downsampling
- 🎛 **Fully connected layers** for classification
- 🚧 **Dropout layers** for regularization
- ✅ The final output layer used **softmax activation** to classify multiple categories of waste, including various plastic types.

### 🎯 Model Training:
- 🏋️ Trained the model for **multiple epochs** using the **Adam optimizer** and **categorical cross-entropy loss function**.
- 📦 Used **batch size of 256** for training.

### 📊 Visualization:
- 🥧 Displayed a **pie chart** showing the distribution of labels in the dataset.
- 🖼 Plotted **random sample images** from the dataset to visually check their labels.
- 📈 Visualized **training and validation loss/accuracy curves**.
- 🔍 Generated a **confusion matrix** to evaluate classification performance.

### 📏 Performance Metrics:
- 📊 Evaluated the model on the **validation set** to track accuracy and loss.
- 📌 Measured **precision, recall, and F1-score** to assess the model’s effectiveness.

---

## 🛠 Technologies Used
- **💻 Programming Language:** Python
- **📚 Libraries/Frameworks:**
  - 🧠 TensorFlow/Keras
  - 📷 OpenCV
  - 🔢 NumPy
  - 📊 Pandas
  - 📉 Matplotlib
  - ⏳ tqdm (for progress bars)

---

## 📂 Dataset
- The dataset consists of images of **various types of waste**, including **plastic, organic, and recyclable materials**.
- 🏷 Each image is **labeled according to its category**.
🔗 The dataset is available at the following link: **[Dataset Link](https://www.kaggle.com/datasets/techsash/waste-classification-data/data)**


---

## ⚙️ Setup Instructions
Follow these steps to run the project:

```bash
# 🚀 Clone the repository
git clone https://github.com/narayana-thota/Cnn-model-to-classify-images-of-plasticwaste.git

# 📁 Navigate to the project directory
cd cnn-model  

# 📦 Install required dependencies
pip install -r requirements.txt  

# 🏋️ Run the model training script
python train.py  

# 📊 Evaluate the model on the test set
python evaluate.py  
