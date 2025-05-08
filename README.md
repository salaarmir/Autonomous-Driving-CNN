# Autonomous Driving with CNNs for Angle and Speed Prediction

This repository contains the code for a convolutional neural network (CNN) project that predicts steering angle and speed for autonomous driving using the MobileNetV2 architecture. Originally developed as part of a university project, it demonstrates the application of biologically inspired neural architectures to real-time perception and decision-making in autonomous systems. This aligns with ongoing research into brain-inspired deep learning and embodied AI.

## Features

* Multi-output CNN for simultaneous angle and speed prediction
* Transfer learning with MobileNetV2
* Real-time processing with TensorFlow Lite
* Image preprocessing and augmentation
* Fine-tuning for improved performance

## Project Structure

```
autonomous-driving-cnn/
├── README.md
├── data/
│   └── example_image.jpg 
├── code/
│   ├── mobilenet_combined.py
│   └── utils.py
├── Project_Report.pdf
├── requirements.txt
```

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/autonomous-driving-cnn.git
cd autonomous-driving-cnn
pip install -r requirements.txt
```

## Usage

Make sure your data is organised correctly, then run the training script:

```bash
python scripts/mobilenet_combined.py
```

## Data Preprocessing

* Converts images to YUV color space (similar to early visual processing)
* Applies Gaussian blur to reduce high-frequency noise (simulating biological edge detection)
* Resizes to 192x192 pixels for efficient computation
* Optional horizontal flipping for data augmentation

## Model Architecture

* Base: MobileNetV2 (pre-trained on ImageNet)
* Separate branches for angle (17 classes, softmax) and speed (binary, sigmoid) outputs
* Uses dropout and batch normalization for regularization

## Training

The models are trained using categorical cross-entropy for angle prediction and binary cross-entropy for speed prediction. Early stopping and model checkpoints are included to improve training stability.

## Results

* **Kaggle Model:** 9th place overall with \~32.67% angle accuracy and \~96.78% speed accuracy
* **Real-time Model:** 4th place in live testing with \~37.38% angle accuracy and \~95.79% speed accuracy

## Future Work

This project could be further improved upon by:

* Implementing synthetic minority over-sampling (SMOTE)
* Testing alternative architectures like Vision Transformers (ViTs)
* Exploring more biologically plausible learning algorithms for real-time control

## License

This project is licensed under the MIT License.

## Acknowledgements

Thanks to the University of Nottingham and the creators of the MobileNetV2 architecture.

## Project Report

For a detailed explanation of the project, please refer to the full [Project Report](Project_Report.pdf).
