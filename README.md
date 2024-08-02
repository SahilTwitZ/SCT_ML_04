# Gesture Recognition Using Convolutional Neural Networks

This project demonstrates a gesture recognition system using Convolutional Neural Networks (CNNs). The model is trained to classify gestures from images, using the Leap Gesture Recognition dataset.

## Project Overview

The goal of this project is to build a CNN model to recognize and classify hand gestures based on images. The model is trained on a dataset of hand gesture images and evaluated for accuracy and performance.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Prerequisites

Make sure you have the following libraries installed:

- `numpy`
- `pandas`
- `PIL` (Pillow)
- `matplotlib`
- `scikit-learn`
- `keras`
- `tensorflow`

You can install them using pip:

```bash
pip install numpy pandas pillow matplotlib scikit-learn keras tensorflow
```

## Dataset

The dataset used in this project is the [Leap Gesture Recognition dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data) available on Kaggle. This dataset includes images of various hand gestures, organized into directories labeled according to the gesture type.

## Project Structure

- `load_data.py`: Contains code to load and preprocess the dataset.
- `model.py`: Contains the definition of the CNN model.
- `train.py`: Contains code to train the model and save the best weights.
- `evaluate.py`: Contains code to evaluate the model on test data.
- `model_architecture.png`: Visual representation of the CNN model architecture.
- `README.md`: This file.

## Model Architecture

The CNN model consists of the following layers:

1. **Conv2D**: 32 filters, (5, 5) kernel size, stride (2, 2), ReLU activation
2. **MaxPooling2D**: (2, 2) pool size
3. **Conv2D**: 64 filters, (3, 3) kernel size, ReLU activation
4. **MaxPooling2D**: (2, 2) pool size
5. **Conv2D**: 64 filters, (3, 3) kernel size, ReLU activation
6. **MaxPooling2D**: (2, 2) pool size
7. **Flatten**
8. **Dense**: 128 units, ReLU activation
9. **Dense**: 10 units, softmax activation (for classification)

## Training and Evaluation

The model is trained with the following settings:

- **Optimizer**: Adam (learning rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 128
- **Epochs**: 100

Early stopping and model checkpointing are used to prevent overfitting and save the best model weights.

## Results

After training, the model's performance is evaluated on test data. The test loss and accuracy are reported.

## Future Work

- Explore data augmentation techniques to improve model robustness.
- Experiment with different CNN architectures and hyperparameters.
- Evaluate the model on additional gesture datasets.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
