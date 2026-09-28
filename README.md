# 🚗 Vehicle Detection using Deep Learning & Image Processing

A Master's semester project focusing on binary image classification for vehicle detection. This project evaluates and compares transfer learning architectures—specifically **InceptionV3** and **Xception**—to accurately classify images into **vehicle** and **non-vehicle** categories.

---

## 🌟 Key Features

* **Dataset Preprocessing & Data Pipeline:** Uses Keras `ImageDataGenerator` for binary classification and automated batch streaming.
* **Transfer Learning Benchmarking:** Comparative analysis between deep learning backbone architectures (**InceptionV3** vs. **Xception**).
* **Comprehensive Model Evaluation:** Tracks Accuracy, Precision, Recall, F1-Score, ROC-AUC, and training time.
* **Visual Analytics:** Complete visualization pipeline featuring loss/accuracy curves and confusion matrices.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python
* **Computer Vision & Image Processing:** OpenCV (`cv2`), PIL
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Processing & Analytics:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---

## 📊 Dataset Structure

The model expects an image dataset structured into binary classes (`vehicles` and `non-vehicles`):

```text
E:/data/
├── non-vehicles/       # Non-vehicle image samples (64x64)
└── vehicles/           # Vehicle image samples (64x64)
