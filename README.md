# 🩺 Chest X-ray Disease Detection (Pneumonia)

This project is a **Deep Learning-based web application** that detects **Pneumonia** from Chest X-ray images using a trained Convolutional Neural Network.

---

## 🚀 Features

* Upload Chest X-ray images (JPG/PNG)
* Predicts whether the patient has **Pneumonia or Normal**
* Displays **prediction confidence**
* Simple and interactive UI using Streamlit

---

## 🧠 Model Details

* Architecture: ResNet18 (Modified)
* Framework: PyTorch
* Type: Binary Classification (Pneumonia vs Normal)
* Output: Probability score

---

## ⚙️ Tech Stack

* Python
* Streamlit
* PyTorch
* Torchvision
* PIL
* gdown (for model download)

---

## 📂 Project Structure

```
📁 chest-xray-app
│-- app.py
│-- requirements.txt
│-- README.md
```

---

## 📥 Model File

Due to GitHub file size limitations, the trained model is hosted on Google Drive.

👉 Download automatically when app runs
Or manually: https://drive.google.com/file/d/1yahPijzUJl-oZ3hB8TvLIMDeQiHWvj6A/view

---

## ▶️ How to Run Locally

1. Clone the repository

```
git clone https://github.com/your-username/chest-xray-app.git
cd chest-xray-app
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the app

```
streamlit run app.py
```

---

## 🌐 Live Demo

(Deploy on Streamlit Cloud and paste your link here)

---

## 📌 Notes

* First run may take time due to model download
* Ensure internet connection is available

---

## ⭐ Acknowledgment

This project is developed for academic and learning purposes in the field of AI & Data Science.
