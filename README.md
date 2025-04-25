# 🌿 Plant Disease Detection Using Convolutional Neural Networks

This project uses a CNN model to detect and classify plant diseases based on leaf images.

## 📌 Overview
- **Goal**: Identify plant diseases using image classification techniques.
- **Tech stack**: Python, OpenCV, Keras, Scikit-learn, Pandas, Matplotlib
- **Dataset**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

## 🧪 Model Summary
- Preprocessing using OpenCV to remove noise.
- CNN architecture with 3 convolutional layers, dropout, and softmax classification.
- Achieved accuracy: **~95%** on validation data.

## 🧠 Features
- Image augmentation and normalization
- Real-time predictions
- Modular training and prediction scripts

## 🔧 How to Run
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the model:
   ```
   python src/train_model.py
   ```
4. Predict on new image:
   ```
   python src/predict.py --image path_to_image.jpg
   ```

## 📂 Folder Structure
- `src/`: All main code files (preprocessing, training, prediction)
- `data/`: Add sample images for testing
- `models/`: Trained models are saved here

## 🤝 Contribution
Feel free to fork and improve!

