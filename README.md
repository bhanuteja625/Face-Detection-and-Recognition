# Deep Learning for Facial Detection and Recognition with PyTorch
This repository contains the code and documentation for a project focused on implementing facial detection and recognition using deep learning techniques. The project leverages the PyTorch library to develop and fine-tune models for these tasks.

## Project Overview
The primary objective of this project is to explore the effectiveness of various deep learning architectures in facial detection and recognition. The project includes:

* Data preprocessing and augmentation techniques
* Model training and evaluation
* Implementation of popular architectures such as YOLOv5 and ResNet
* Analysis of model performance on different datasets

  
## Key Features
1. Data Preprocessing: Techniques for handling different image formats and applying necessary augmentations to enhance model robustness.
2. Model Architecture: Implementation of YOLOv5 for face detection and ResNet for face recognition, with pre-trained weights for efficient training.
3. Training and Evaluation: Comprehensive training scripts with hyperparameter tuning and evaluation metrics to assess model performance.
4. Inference: Scripts for real-time facial detection and recognition from live video feeds or static images.

   
## Repository Structure
* data/: Contains datasets and data processing scripts.
* models/: Pre-trained models and scripts for training new models.
* notebooks/: Jupyter notebooks for exploratory data analysis and model prototyping.
* utils/: Utility functions and helper scripts.
  
## Getting Started
## Prerequisites
* Python 3.x
* PyTorch
* OpenCV
* Other dependencies listed in requirements.txt
  
## Installation
Clone the repository:
  ```
  git clone https://github.com/bhanuteja625/Face-Detection-and-Recognition.git
  cd deep-learning-facial-detection-recognition
  ```

Install the required packages:

    pip install -r requirements.txt


## Usage

### Jupyter Notebooks
Explore the model's training and inference processes using the provided Jupyter notebook:

1. Open the notebooks/ directory and start the notebook model.ipynb.
2. Follow the cells to understand data preprocessing, model training, and evaluation.
3. Use the downloaded model for inference by loading it and passing your data through the inference pipeline.
Real-Time Inference with Flask

### For real-time facial recognition, use the app.py script:

1. Ensure you have all the necessary dependencies installed.
2. Start the Flask app by running:
   
   ```
   python app.py
   ```
   
3. The application will be accessible at http://127.0.0.1:5000/. Use the web interface to upload images or access a live video feed for real-time facial detection and recognition.
   
## Results
The project report contains detailed analysis and comparison of model performances on various datasets. Key metrics such as accuracy, precision, recall, and F1-score are provided to evaluate the effectiveness of the models.


## Acknowledgements
PyTorch for providing the deep learning framework.
The creators of YOLOv5 and ResNet for their contributions to the field.
