# Facial Recognition CNN - Blink Detection

A deep learning project that uses Convolutional Neural Networks (CNN) to detect eye blinks in real-time using facial landmarks and computer vision techniques.

## Project Overview

This project implements an eye blink detection system using a trained CNN model. The system:
- Detects faces in video frames using Haar Cascade classifiers
- Extracts facial landmarks to locate eyes precisely
- Crops eye regions to a standardized size (34x26 pixels)
- Uses a CNN model to classify eyes as open or closed
- Counts blinks in real-time with audio feedback

## Features

- **Real-time Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces
- **Facial Landmark Detection**: Leverages dlib to identify 68 facial landmarks
- **Eye Cropping**: Automatically extracts and resizes eye regions for CNN input
- **CNN-based Classification**: Binary classification of eye state (open/closed)
- **Blink Counting**: Tracks consecutive blinks with configurable thresholds
- **Audio Feedback**: Plays audio alerts when blinks are detected
- **Data Augmentation**: Uses Keras ImageDataGenerator for robust model training

## Project Structure

```
.
├── README.md                    # This file
├── eyes with tensor.ipynb       # Model training notebook
├── cnn test-2.ipynb            # Real-time blink detection notebook
└── blinkModel.hdf5             # Trained CNN model (not included)
```

## Requirements

- Python 3.7+
- OpenCV (cv2)
- Keras with TensorFlow backend
- NumPy
- Scipy
- dlib
- imutils
- pygame
- Jupyter Notebook

### Installation

```bash
pip install opencv-python keras tensorflow numpy scipy dlib imutils pygame
```

## Model Architecture

The CNN model consists of:
- **Input Layer**: 26x34x1 (grayscale eye images)
- **Conv2D Layers**: 
  - 32 filters (3x3 kernel)
  - 64 filters (2x2 kernel)
  - 128 filters (2x2 kernel)
- **MaxPooling Layers**: (2x2) after each convolutional block
- **Dense Layers**:
  - 512 units (ReLU activation)
  - 512 units (ReLU activation)
  - 1 unit (Sigmoid activation for binary classification)
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Binary crossentropy

## Training

Training is performed in `eyes with tensor.ipynb`:

1. Load eye image dataset from CSV file
2. Normalize pixel values to [0, 1]
3. Apply data augmentation (rotation ±10°, width/height shifts ±20%)
4. Train for 50 epochs with batch size of 32
5. Save trained model as `blinkModel.hdf5`

**Dataset Format**: CSV with columns:
- `image`: Flattened eye image as a list of pixel values
- `state`: Label ('open' or 'closed')

## Usage

### Real-time Blink Detection

Run `cnn test-2.ipynb` to start real-time blink detection:

```python
# Key components:
camera = cv2.VideoCapture(0)
model = load_model('blinkModel.hdf5')

# Main loop processes video frames:
# 1. Detects faces and extracts eyes
# 2. Predicts eye state using CNN
# 3. Counts blinks when eyes close then open
# 4. Displays blink count and eye state on video
```

**Controls**:
- Press 'q' to quit the application

**Output**:
- Real-time video display with:
  - Blink counter
  - Current eye state (open/closed)

## Key Functions

### `detect(img, cascade, minimumFeatureSize=(20, 20))`
Detects faces using Haar Cascade classifier and returns bounding boxes.

### `cropEyes(frame)`
Extracts left and right eye regions from a face:
- Converts frame to grayscale
- Detects face using Haar Cascade
- Uses dlib's 68-point facial landmark detector
- Crops eyes to (34, 26) standard size
- Returns None if face or eyes not detected

### `cnnPreprocess(img)`
Prepares eye images for CNN input:
- Converts to float32
- Normalizes to [0, 1]
- Expands dimensions for channel and batch

### `readCsv(path)`
Loads training data from CSV format:
- Converts string representations back to image arrays
- Assigns labels (1 for open, 0 for closed)
- Shuffles dataset

### `makeModel()`
Constructs and compiles the CNN architecture.

## Parameters

- **Eye Image Size**: 34 x 26 pixels
- **Minimum Face Size**: (80, 80)
- **Blink Threshold**: 2+ consecutive frames of closed eyes
- **Model Prediction Threshold**: 0.5 (>0.5 = open, ≤0.5 = closed)
- **Training Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 0.001

## Performance

The model achieves ~95% accuracy on the training dataset:
- Epoch 1: 39.6% accuracy
- Epoch 50: 95.1% accuracy

## Limitations

- Requires good lighting conditions for accurate face detection
- Performance depends on face orientation and distance from camera
- Currently uses Keras/TensorFlow backend (deprecated in newer TensorFlow versions)
- Audio file path is hardcoded ('no-3-Copy1.wav')
- Dataset path is hardcoded in training notebook

## Future Improvements

- Migrate to TensorFlow 2.x with Keras API
- Support multiple faces in a single frame
- Add configurable parameters
- Implement model conversion to TFLite for mobile deployment
- Add gaze direction detection
- Enhance robustness with head pose estimation
- Create standalone Python script (non-notebook)

## Dependencies Attribution

- **OpenCV**: Real-time computer vision library
- **dlib**: Face detection and landmark localization
- **Keras/TensorFlow**: Deep learning framework
- **imutils**: Utility functions for computer vision
- **Pygame**: Audio playback

## Notes

- The trained model `blinkModel.hdf5` is required for inference
- Training requires a labeled dataset in the specified CSV format
- Ensure webcam is connected and accessible for real-time detection
- The project uses deprecated Keras backend (consider updating)

## License

This project is provided as-is for educational purposes.

## Author

Tanishq Cruz (@cruzTanishq)
