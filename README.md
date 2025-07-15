# ✋ SkillCraft Task 4 – Hand Gesture Recognition

A deep learning-based project to recognize and classify hand gestures using a Convolutional Neural Network (CNN). The system enables intuitive Human-Computer Interaction (HCI) through gesture-based control using either **uploaded images** or **live camera input**.

---

## 📁 Dataset

- **Name:** [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Source:** Kaggle
- **Classes:** 10 gesture categories:
  - `00_palm`
  - `01_l`
  - `02_fist`
  - `03_fist_moved`
  - `04_thumb`
  - `05_index`
  - `06_ok`
  - `07_palm_moved`
  - `08_c`
  - `09_down`
- **Total Images:** 20,000+
- **Format:** JPEG (directory-based)

---

## 🧠 Model Architecture

Implemented a CNN using TensorFlow/Keras:

text
Input (128x128x3) ➝ Conv2D ➝ MaxPooling ➝ BatchNorm ➝
Conv2D ➝ MaxPooling ➝ BatchNorm ➝
Conv2D ➝ MaxPooling ➝ BatchNorm ➝
Flatten ➝ Dense (256) ➝ Dropout ➝ Output (Softmax)
Activation: ReLU

Final Layer: 10-class Softmax

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Accuracy Achieved: ~97% on validation

🧪 Project Files
File Name	Purpose
hand-gesture-recognition.ipynb	Model training notebook in Kaggle
gesture_cnn_model.h5	Saved CNN model
X_val.npy, Y_val.npy	Preprocessed test data
app.py	Flask API backend
index.html	Frontend with file upload + camera input
static/uploads/	Directory for temporary uploaded files

🚀 Features
🔍 Real-time prediction via image upload or webcam

🌐 Interactive web interface using HTML, CSS, and JavaScript

🌗 Dark/Light mode toggle

📸 Live camera support (with capture)

📊 High accuracy with intuitive prediction result display

🖥️ How to Run Locally
⚠️ Requires Python ≥ 3.8 with TensorFlow, Flask, and other packages.

1. Clone Repository & Install Dependencies
bash
Copy
Edit
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
pip install -r requirements.txt
2. Run Flask App
bash
Copy
Edit
python app.py
3. Open in Browser
Go to: http://127.0.0.1:5000

🧩 Tech Stack
Frontend: HTML5, CSS3, JavaScript

Backend: Flask (Python)

Model: CNN (TensorFlow/Keras)

Deployment: Localhost / Flask Dev Server

📷 Demo
You can either:

Upload an image of a hand gesture

Or use your webcam to capture live

Prediction will appear after clicking the Predict button.

🏁 Output Example
Input Image	Predicted Gesture
Palm	00_palm
Thumb Up	04_thumb
OK Sign	06_ok

🏆 Internship Project
This project was developed as part of SkillCraft Machine Learning Internship – Task 4: Gesture Recognition.

📬 Contact
For queries or collaborations:

👩 Medha Jha
📧 LinkedIn 
🔗 GitHub Profile : https://github.com/medhajha810/

