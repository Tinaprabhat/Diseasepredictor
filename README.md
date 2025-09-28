

# 🧠 ML-Based Disease Predictor

A **Machine Learning-powered Disease Prediction System** that uses patient health data to classify and predict diseases.
The system is trained on a dataset downloaded from **Kaggle**, and implements multiple classification algorithms to select the best-performing model for final predictions.

---

## ⚡ Features

* ✅ Trained on real-world Kaggle dataset
* ✅ Implements **3 classification models**:

  * Support Vector Machine (SVM)
  * Naive Bayes
  * Random Forest
* ✅ Automatically selects the **best model** based on accuracy
* ✅ Simple interface for making predictions

---

## 📂 Project Structure

```
disease-predictor/
│-- diseasepredictmodel.py     # Main ML training & prediction script
│-- UIdiseasepredictor.py                   # UI (Streamlit/Flask, if applicable)
│-- dataset.csv              # Kaggle dataset (not included in repo if too large)
│-- requirements.txt         # Dependencies
│-- README.md                # Documentation
│-- .gitignore               # Ignored files
```

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/disease-predictor.git
cd disease-predictor
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train and evaluate models:

```bash
python disease_predictor.py
```

The script will:

* Preprocess the Kaggle dataset
* Train **SVM, Naive Bayes, and Random Forest** models
* Compare accuracies and select the **best model**

### Run the predictor (if UI provided):

```bash
streamlit run app.py
```

Then open the local server link in your browser (usually `http://localhost:8501`).

---

## 📦 Requirements

Add these to `requirements.txt`:

```
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
```

---

## 📊 Model Evaluation

During training, the system evaluates each classifier:

| Model         | Accuracy |
| ------------- | -------- |
| SVM           |  60.53%  |
| Naive Bayes   |   37.98% |
| Random Forest | 68.98%   |

✅ The best model (highest accuracy) is used for predictions.

---

## 🛠️ Tech Stack

* **Python 3.12+**
* **Pandas & NumPy** → Data handling
* **Scikit-learn** → ML models (SVM, Naive Bayes, Random Forest)
* **Matplotlib & Seaborn** → Data visualization
* **Streamlit** (optional) → Interactive UI

---

## 🤝 Contributing

Pull requests are welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Added new feature"`)
4. Push to branch (`git push origin feature-name`)
5. Create a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** – you are free to use and modify it.
