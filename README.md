# 💳 AI Loan Approval System (Decision Tree)

🔗 **Live Demo (Hugging Face Space):**
https://huggingface.co/spaces/hishaaaam/Loan-Approval-usingTree

An end-to-end machine learning application that predicts whether a loan application should be **Approved or Rejected** using a **Decision Tree Classifier**. The system features a modern Gradio interface for real-time user interaction and is designed to be lightweight, fast, and deployable.

---

## 🚀 Overview

This project demonstrates a complete ML deployment pipeline:

* 🌳 Decision Tree classification model
* 🧠 Financial risk assessment logic
* 🎨 Professional Gradio UI
* ⚡ Real-time predictions
* ☁️ Hugging Face Spaces deployment

The model analyzes applicant financial attributes and outputs an automated credit decision along with a confidence score.

---

## ✨ Features

* ✅ Real-time loan approval prediction
* 📊 Confidence score visualization
* 🎨 Modern responsive UI
* ⚡ Fast inference
* 🧩 Single-file deployable architecture
* ☁️ Hugging Face Spaces ready
* 🛡️ Input validation and error handling

---

## 🧠 Model Details

**Algorithm:** Decision Tree Classifier
**Problem Type:** Binary Classification

### 📥 Input Features

* Annual Income
* Credit Score
* Loan Amount
* Age
* Employment Years

### 📤 Output

* ✅ Approved
* ❌ Rejected
* 📊 Confidence Score

---

## 🛠️ Tech Stack

* **Machine Learning:** Scikit-learn
* **Frontend/UI:** Gradio
* **Language:** Python
* **Model Serialization:** Joblib
* **Deployment:** Hugging Face Spaces

---

## 📂 Project Structure

```
.
├── app.py
├── requirements.txt
└── model/
    └── model.pkl
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2️⃣ Create virtual environment (recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

Then open the local Gradio link shown in the terminal.

---

## ☁️ Deployment (Hugging Face Spaces)

This project is fully compatible with **Gradio Spaces**.

**Steps:**

1. Create a new Space (SDK: Gradio)
2. Upload:

   * `app.py`
   * `requirements.txt`
   * `model/model.pkl`
3. The Space will automatically build and run.

---

## 🧪 Example Test Cases

### ✅ Expected Approval

```
Income: 90000  
Credit Score: 720  
Loan Amount: 20000  
Age: 30  
Employment Years: 5
```

### ❌ Expected Rejection

```
Income: 30000  
Credit Score: 520  
Loan Amount: 30000  
Age: 24  
Employment Years: 0
```

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**.
It should **not** be used as a real financial approval system.

---

## 👨‍💻 Author

**Hisham Hidayathulla**

If you found this project useful, consider giving it a ⭐ on GitHub!
