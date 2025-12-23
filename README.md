# skin-cancer-diagnosis
### Diagnosing Skin Cancer with Machine Learning

The goal of this project is to research and evaluate machine learning models that can
assist with early skin cancer screening via a web application. This project explores computer vision, classical ML, deep learning, transfer learning, and model evaluation techniques, with an emphasis on accessibility and protoyping real-world deployment.

## File Structure
The repository is organized to separate data handling, model development, evaluation, and training workflows.

```text
skin-cancer-diagnosis/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   │   └── Dataset loading, splitting, and label handling
│   │   ├── preprocessing.py
│   │   │   └── Image normalization and one-hot encoding
│   │   └── augmentation.py
│   │       └── OpenCV-based image augmentation utilities
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   │   └── Custom CNN architecture (SciKeras-compatible)
│   │   └── transfer_learning.py
│   │       └── VGG16-based transfer learning model
│   │
│   ├── training/
│   │   ├── train_cnn.py
│   │   │   └── Training pipeline for custom CNN
│   │   └── train_transfer.py
│   │       └── Training pipeline for transfer learning model
│   │
│   ├── eval/
│   │   └── metrics.py
│   │       └── Accuracy, ROC-AUC, and confusion matrix utilities
│   │
│   └── web/
│       ├── app.py
│       │   └── Local web app for image upload and inference
│       ├── templates/
│       │   └── HTML templates
│       └── static/
│           ├── css/
│           │   └── UI styling
│           └── js/
│               └── Client-side behavior
│
├── requirements.txt
│   └── Python dependencies required to run the project
│
└── README.md
    └── Project overview, methodology, and usage instructions
```
## Background
Skin cancer is the **most common form of cancer worldwide**, accounting for a significant portion of all cancer diagnoses. Early detection is critical for successful treatment, yet many individuals lack consistent access to professional dermatological care. By taking the diagnosis process to a web-based format, this project aims to protoype a system that would serve those in rural or underserved communities where traditional diagnosis may have additional barriers.

### Key Challenges Addressed

1. Limited access to trained dermatologists and medical imaging equipment

2. Variability in image quality when photos are taken on consumer smartphones

3. Bias and generalization issues in medical ML models

4. Scalable, low-cost screening options

### Existing Technologies and Limitations

1. MelaFind – FDA-approved diagnostic support tool for melanoma detection

2. SkinVision – Consumer-facing mobile app for skin cancer risk assessment

This project explores how modern machine learning techniques can complement existing tools by providing a lightweight, accessible screening solution

## Data

### Skin Cancer MNIST: HAM10000 Dataset
- Sourced from Harvard University’s Dataverse Repository (also available on Kaggle)
- Over 10,000 dermatoscopic images of pigemented lesions
- Images comprise 7 diagnostic classes:
  1. Melanocytic nevus
  2. Melanoma
  3. Benign keratosis
  4. Basal cell carcinoma
  5. Actinic keratosis
  6. Dermatofibroma
  7. Vascular lesions

### Data Challenges
- Highly imbalanced class distribution
- Images captured using dermatoscopes, unlike real-world smartphone photos
- High dimensionality of raw image data

### Data Augmentation Techniques (OpenCV)
To improve generalization and simulate real-world conditions, augmentation was applied only to training data, including:
- Horizontal and vertical flips
- Gaussian blurring
- Image resizing
- Zoom and crop transformations

### Dataset Setup

The HAM10000 images are loaded locally. In order to re-train the model, take the following steps to 
setup the dataset:
1. Download from Kaggle:
   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Place images in:
   raw-data/images/
3. Place metadata file in:
   data/HAM10000_metadata.csv

## Machine Learning Models

All models were implemented in **Python**, primarily using [`scikit-learn`](https://scikit-learn.org/stable/developers/index.html), [`TensorFlow`](https://www.tensorflow.org/guide), and [`Keras`](https://keras.io/guides/).

### Classical Machine Learning
- **K-Nearest Neighbors (KNN)**
  - Trained on flattened grayscale images
  - Used as a baseline classifier
  - Demonstrated limitations when applied to high-dimensional image data

### Deep Learning
- **Convolutional Neural Network (CNN)**
  - Custom architecture built with Keras
  - Multiple convolutional, pooling, and dropout layers

### Transfer Learning
- **VGG16 (Keras Applications)**
  - Pretrained on ImageNet
  - Final classification layers replaced and retrained on skin lesion data
  - Achieved the strongest overall performance

## Evaluation & Metrics

Given the imbalanced nature of the dataset, multiple evaluation metrics were used:
- **Receiver Operating Characteristic (ROC)** curves
- **Area Under the Curve (AUC)**
- **Confusion matrices** for class-level performance analysis

### Key Findings
- Accuracy alone was misleading due to class imbalance
- ROC/AUC provided a more reliable indicator of performance
- **Best-performing model**: Transfer Learning with VGG16

## Bias & Model Limitations

During evaluation, the transfer learning model initially:
- Performed noticeably worse on images of darker skin tones
- Reflected known limitations of medical imaging datasets and model bias

To address this:
- Training data was expanded to include greater variation in skin tones
- Model behavior was analyzed to identify and reduce performance disparities

This component of the project emphasizes the importance of **ethical and responsible machine learning**, particularly in healthcare contexts.

## Deployment

The final model was deployed as a **web-based application**, allowing users to interact with the system without specialized software.

### Technologies Used
- **HTML & JavaScript** for the frontend
- Python-based backend serving model predictions
- Image upload, preprocessing, and inference pipeline

## Results & Impact

The deployed web application enables users to:
- Upload an image of a skin lesion
- Receive a predicted diagnosis
- View confidence scores associated with the prediction

### Outcomes
- Successfully built a **proof-of-concept diagnostic screening tool**
- Demonstrated the feasibility of ML-based skin lesion classification
- Created a **user-friendly and accessible interface**
- Highlighted potential benefits for individuals lacking regular access to dermatological care

## Future Work & Extensions

Planned improvements and extensions include:
- Further optimization of the transfer learning model
- Exploration of additional architectures (e.g., ResNet, EfficientNet)
- Development of a **mobile application**
- Real-time lesion screening using smartphone cameras
- Continued focus on fairness, bias mitigation, and dataset diversity

The long-term goal is to create a **highly accessible, ethical, and reliable skin cancer screening aid**.

<br>

---

⚠️ **Note:**  
This project is for educational purposes only and serves as a proof-of-concept for a more accessible system. It is not meant to be used as a professional medical tool.
