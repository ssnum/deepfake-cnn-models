# CNN Preprocessing and Model Implementations for Deepfake Detection Research

This repository contains the code for preprocessing my dataset and implementing various Convolutional Neural Network (CNN) architectures, including VGG16, VGG19, DenseNet121, ResNet50, along with custom models and an age classifier. The code is designed to facilitate my independent research on deepfake detection.

## Features

- **Data Preprocessing:** Code to download and preprocess datasets for training and testing
- **Model Implementations:**
  - VGG16 (Built layers from scratch): Standard implementation for image classification tasks.
  - VGG19 (Built layers from scratch): Similar to VGG16 with more layers for deeper feature extraction.
  - VGG Transfer Learning : Used transfer learning instead of building layers from scratch
  - DenseNet121: Efficient model that connects layers in a dense fashion to improve gradient flow.
  - ResNet50: Residual network that allows for very deep architectures using skip connections.
  - Custom Models: Various experimental models created to test different architectural features.
  - Age Classifier Models: Created 12 Specialized models aimed at predicting the age of subjects in images and applied it to my data. Taken from a previously conducted study and modified to fit my data.
  - Printing and saving results for all 12 models from Age Classifier Experiment including: best and worst model confusion matrix, all 12 validation curves

## Requirements to Use

- Must import all necessary libraries as referenced in the code, including TensorFlow, Keras, NumPy, and Matplotlib.
- Ensure you have the datasets downloaded from Kaggle or any other source before running the preprocessing scripts.

## Credits

- Original Models and Implementations from Keras: [Keras Documentation](https://keras.io/)
- Datasets referenced from Kaggle: [Deepfake Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Age Classification Using an Optimized CNN Architecture by M. Fatih Aydogdu and M. Fatih Demirci : https://drive.google.com/file/d/1YeDgxSu134cg0UzsIEnCMi4np5OC-0wf/view?usp=sharing 
