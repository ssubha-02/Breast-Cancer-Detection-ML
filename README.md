# 🩺 Breast Cancer Detection using Machine Learning  
*A complete ML pipeline + Streamlit Web App*

---

## 📌 Project Overview  

This project builds a Machine Learning model to classify breast tumors as **Benign (0)** or **Malignant (1)** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.  

It includes:

- Full ML training pipeline (cleaning → scaling → training → evaluation)  
- Two ML models (Logistic Regression & Random Forest)  
- Saving the best model using Joblib  
- A fully interactive **Streamlit web app** for real-time prediction  
- A clean, readable, beginner-friendly code structure  

🔬 **Best Model:** Random Forest  
🎯 **Accuracy:** ~96%  
📊 **Precision for Malignant:** 100%  

---

## 📁 Folder Structure  

```
Breast-Cancer-Detection-ML
│
├── data.csv                      # dataset
├── main.py                       # training pipeline
├── app.py                        # Streamlit web app
├── model_scaler.joblib           # saved scaler
├── model_random_forest.joblib    # saved ML model
├── venv/                         # virtual environment
└── README.md                     # project documentation
```

---

## 🧠 Dataset Information  

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
- **Source:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  
- **Samples:** 569  
- **Features:** 30 continuous tumor measurement features  
- **Target:**  
  - `B` → Benign (0)  
  - `M` → Malignant (1)  

---

## 🚀 Model Training (main.py)

### ML Pipeline Includes:
✔ Loading dataset  
✔ Dropping unwanted columns (`id`, `Unnamed: 32`)  
✔ Encoding labels (`B` → 0, `M` → 1)  
✔ Train-test split (80/20)  
✔ Feature scaling  
✔ Training two models:  
   - Logistic Regression  
   - Random Forest  
✔ Evaluating using Accuracy, Precision, Recall, F1-score, ROC-AUC  
✔ Saving the best model  

Run training script:

```bash
python main.py
```

This generates:  
- `model_scaler.joblib`  
- `model_random_forest.joblib`  

---

## 🌐 Streamlit Web App (app.py)

The Streamlit app allows users to input tumor measurement values and instantly get:

- Prediction → **Benign or Malignant**
- Probability score
- Clean UI + user-friendly form

### Run the web app:

```bash
streamlit run app.py
```

This opens your app at:

```
http://localhost:8501
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd Breast-Cancer-Detection-ML
```

### 2️⃣ Create & activate virtual environment

#### Windows (CMD):
```bash
python -m venv venv
venv\Scripts\activate.bat
```

#### Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install streamlit pandas scikit-learn numpy joblib matplotlib seaborn
```

---

## 📊 Model Performance (Summary)

### Logistic Regression
- Accuracy: ~96%  
- Malignant Recall: ~92%  
- Malignant Precision: ~97%  

### Random Forest (Best Model)
- Accuracy: ~96%  
- Malignant Precision: **100%**  
- Malignant Recall: ~90%  

---

## 🧪 Example Prediction Output (from Streamlit)

- **Prediction:** Malignant  
- **Probability:** 89.42%  

Or:

- **Prediction:** Benign  
- **Probability:** 7.31%  

---

## 🔮 Possible Future Improvements

- Add SHAP explainability  
- Add confusion matrix & feature importance visualization  
- Deploy on Streamlit Cloud  
- Build REST API using FastAPI or Flask  
- Create a mobile-friendly UI  

---

## 🙌 Acknowledgements  

Dataset by:  
**University of Wisconsin Hospitals, Madison**  
Available on Kaggle

---

## 📝 License  
This project is for **educational and research purposes only**.  
Not to be used for real medical diagnosis.

---

# 🎉 Thank you for exploring this project!
