# 🩺 SkinGuard AI

<div align="center">

# Advanced Skin Disease Detection & DrugGPT Healthcare Assistant

AI-powered skin disease detection system built with Deep Learning, Computer Vision, and an intelligent medical assistant.

🌐 **Live Demo:** https://huggingface.co/spaces/adalis/SkinGuard-AI-Skin-Disease-Detection-System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![Healthcare](https://img.shields.io/badge/AI-Healthcare-green)
![License](https://img.shields.io/badge/License-MIT-purple)

</div>

---

# 📖 Overview

SkinGuard AI is an intelligent healthcare platform that combines state-of-the-art deep learning models with an AI-powered medical assistant to analyze skin images and provide educational insights about skin diseases.

The system can identify multiple dermatological conditions, provide confidence scores, explain symptoms and causes, suggest educational treatment information, and offer prevention recommendations through DrugGPT.

---

# ✨ Key Features

## 🔍 AI Skin Disease Detection

* Deep learning-based image classification
* Real-time disease prediction
* Confidence score visualization
* 20 supported skin disease categories
* Fast inference and analysis

## 🤖 DrugGPT Assistant

* Disease explanations
* Symptom analysis
* Treatment information
* Medicine knowledge base
* Prevention guidance
* Doctor consultation recommendations

## 📊 Clinical Insights

* Disease severity assessment
* Contagiousness information
* Home-care recommendations
* Risk factor analysis

## 🎨 Modern Interface

* Responsive design
* Dark futuristic UI
* Mobile-friendly experience
* Interactive dashboards

---

# 📊 Model Performance

## 🧠 Architecture

| Metric              | Value               |
| ------------------- | ------------------- |
| Model Backbone      | EfficientNet-B4     |
| Classification Head | Custom Dense Layers |
| Framework           | PyTorch             |
| Disease Classes     | 20                  |
| Dataset Size        | 10,000 Images       |
| Deployment          | Hugging Face Spaces |

---

## 📈 Performance Metrics

| Metric        | Score  |
| ------------- | ------ |
| Test Accuracy | 92.7%  |
| Precision     | 92.76% |
| Recall        | 92.73% |
| Classes       | 20     |
| Total Images  | 10,000 |

---

## 📂 Dataset Split

| Dataset    | Images |
| ---------- | ------ |
| Training   | 7,000  |
| Validation | 1,500  |
| Testing    | 1,500  |

---

# 📷 Training Dashboard

![Training Dashboard](assets/training_dashboard.png)

The model demonstrates stable convergence with strong validation performance and minimal overfitting.

---

# 🩺 Supported Diseases

## 🔥 Inflammatory Conditions

* Acne & Rosacea
* Atopic Dermatitis (Eczema)
* Skin Rashes
* Urticaria (Hives)
* Vasculitis

## 🦠 Bacterial Diseases

* Cellulitis
* Impetigo

## 🍄 Fungal Diseases

* Athlete's Foot
* Ringworm
* Fungal Nail Infection
* Nail Diseases

## 🧬 Viral Diseases

* Chickenpox
* Shingles
* Herpes
* HPV Related Conditions

## 🪱 Parasitic Diseases

* Cutaneous Larva Migrans

## 🎗️ Skin Cancer & Lesions

* Melanoma
* Malignant Skin Lesions
* Benign Skin Lesions
* Moles & Nevi

## 💇 Hair Disorders

* Alopecia
* Hair Loss Disorders

## ✅ Healthy Skin Detection

* Normal Skin Classification

---

# 🤖 DrugGPT Healthcare Assistant

DrugGPT transforms SkinGuard AI from a simple image classifier into a comprehensive healthcare education platform.

## DrugGPT Features

### 📚 Disease Knowledge

* Disease overview
* Symptoms
* Causes
* Risk factors
* Severity levels

### 💊 Treatment Guidance

* Topical treatments
* Systemic treatments
* Home-care recommendations
* Prevention methods

### 🚨 Medical Guidance

* Doctor visit recommendations
* Critical warning signs
* Disease monitoring information

---

# 💊 Medicine Knowledge Base

DrugGPT includes educational information about commonly used treatments.

### Acne & Rosacea

* Benzoyl Peroxide
* Adapalene
* Azelaic Acid
* Metronidazole
* Isotretinoin

### Eczema

* Hydrocortisone
* Tacrolimus
* Pimecrolimus
* Antihistamines

### Fungal Diseases

* Clotrimazole
* Terbinafine
* Miconazole

### Viral Diseases

* Acyclovir
* Valacyclovir

### Allergic Conditions

* Antihistamines
* Corticosteroids

> DrugGPT provides educational information only and does not prescribe medications.

---

# 🏗️ Project Structure

```text
SkinGuard-AI/
│
├── app.py
├── model.py
├── best_model.pth
├── class_mapping.json
├── druggpt_engine.py
├── medicine_database.json
├── skin_disease_data.py
├── training_metrics.json
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/SkinGuard-AI.git
cd SkinGuard-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

# 🧠 How It Works

1. Upload a skin image.
2. The image is preprocessed.
3. EfficientNet-B4 analyzes visual patterns.
4. The model predicts the most likely disease.
5. Confidence scores are generated.
6. Disease information is displayed.
7. DrugGPT provides educational treatment guidance.

---

# 🎯 Applications

* AI Healthcare Research
* Medical Education
* Dermatology Demonstrations
* Deep Learning Projects
* Computer Vision Research

---

# ⚠️ Disclaimer

SkinGuard AI and DrugGPT are intended for educational, research, and demonstration purposes only.

The application does not provide medical diagnoses, prescriptions, or professional healthcare advice. Predictions may contain inaccuracies and should never replace consultation with qualified healthcare professionals.

Always seek advice from licensed medical practitioners regarding diagnosis, treatment, and medication decisions.

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

<!--# 📜 License

MIT License

--->

<div align="center">

### 🩺 SkinGuard AI

**Detect • Analyze • Learn • Prevent**

Built with ❤️ using Python, PyTorch, Gradio, and AI.

</div>
