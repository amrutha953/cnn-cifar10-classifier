# cnn-cifar10-classifier
📦 CIFAR-10 CNN Classifier with Data Augmentation

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras. It includes data augmentation, model training, and evaluation with accuracy/loss plots and a confusion matrix.

📌 Dataset

The model is trained and evaluated on the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

Classes:

['airplane', 'automobile', 'bird', 'cat', 'deer',
 'dog', 'frog', 'horse', 'ship', 'truck']

🚀 Features

CNN-based image classifier using TensorFlow/Keras.

Real-time data augmentation for improved generalization.

Training/Validation accuracy and loss plots.

Confusion matrix for evaluating classification performance.

🧪 Model Architecture
Input: 32x32 RGB image
↓ Conv2D (32 filters) + ReLU
↓ MaxPooling2D
↓ Conv2D (64 filters) + ReLU
↓ MaxPooling2D
↓ Conv2D (64 filters) + ReLU
↓ Flatten
↓ Dense (64 units) + ReLU
↓ Dropout (0.5)
↓ Dense (10 units) — Output layer

⚙️ Requirements

Make sure you have the following packages installed:

pip install tensorflow matplotlib scikit-learn numpy


Or use a requirements.txt file if you'd like:

tensorflow
matplotlib
scikit-learn
numpy

🛠️ How to Run

Clone the repository or copy the script.

Install the required dependencies.

Run the Python script:

python cifar10_cnn.py
