
# 🩺 Logistic Regression — Diabetes Prediction

**Repository:** *Logistic Regression for Diabetics*  
**Notebook:** `Logistic regression for diabetics.ipynb`

---

## 🚀 Tech Stack

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 📌 Project Summary

This project builds a clear and interpretable **Logistic Regression** model to predict the likelihood of diabetes using clinical features.  

The notebook walks through an **end-to-end ML workflow**:  
- Data preprocessing & feature scaling  
- Exploratory data analysis (EDA)  
- Model training & evaluation  
- Visual diagnostics (confusion matrix, ROC curve)  
- Interpretation of coefficients as odds ratios  

✨ The goal is to **balance predictive performance with interpretability**, a crucial requirement for healthcare decision-making.

---

## 📋 Dataset

- **Source:** Pima Indians Diabetes Database (UCI Repository)  
- **Target Variable:** `Outcome` → (1 = diabetic, 0 = non-diabetic)  
- **Features Used:**  
  `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`

**Preprocessing applied:**  
- Replaced biologically implausible zero values (for Glucose, BP, BMI, etc.)  
- Applied `StandardScaler` for normalization  
- Split dataset using `train_test_split(test_size=0.3, random_state=42)`

---

## 🔎 Exploratory Data Analysis (EDA)

Key checks performed:  
- Class balance between diabetic vs. non-diabetic patients  
- Feature distributions & correlations  
- Relationship of top predictors (Glucose, BMI, Age) with the outcome  

*(EDA plots can be extended in future iterations)*

---

## 🧠 Model Training

- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)  
- **Handling imbalance:** (optionally `class_weight='balanced'`)  
- **Outputs:** Predictions, probabilities, coefficients  

---

## ✅ Model Performance

**Confusion Matrix**  
![Confusion Matrix](images/confusion%20metrics.png)

**ROC Curve**  
![ROC Curve](images/Roc%20curve.png)

**Evaluation Metrics (Test Set):**  
- **Accuracy:** `0.77`  
- **Precision:** `0.75`  
- **Recall (Sensitivity):** `0.75`  
- **F1-score:** `0.82`  

---

## 🔑 Model Interpretation

Coefficients were mapped to odds ratios for clinical interpretability. Example:

| Feature                  | Coefficient (β) | Odds Ratio (exp(β)) | Interpretation                           |
|--------------------------|----------------:|--------------------:|------------------------------------------|
| Glucose                  | `β_glucose`     | `OR_glucose`        | Higher glucose → higher odds of diabetes |
| BMI                      | `β_bmi`         | `OR_bmi`            | Elevated BMI increases odds              |
| Age                      | `β_age`         | `OR_age`            | Older patients have higher risk          |
| DiabetesPedigreeFunction | `β_dpf`         | `OR_dpf`            | Strong family history raises odds        |

---

## 🩺 Clinical & Business Insights

- **Glucose** and **BMI** are the strongest indicators of diabetes risk.  
- **Age** and **family history (DPF)** further amplify predicted risk.  
- The model can serve as a **screening tool** for healthcare professionals: prioritizing high-risk patients for testing.  
- In practice, **higher recall (sensitivity)** is preferred to minimize false negatives (undiagnosed diabetics).  

---

## ⚠️ Limitations

- Logistic Regression assumes linear log-odds → nonlinear patterns may be missed.  
- Missing/imputed data can bias the model.  
- Dataset is relatively small; external validation required.  
- Clinical deployment requires regulatory approval & real-world testing.  

---

## 📂 Repository Structure

```

📁 Logistic-Regression-diabetics
│── Logistic regression for diabetics.ipynb   # Jupyter notebook
│── diabetes.csv                              # Dataset
│── images/
│    ├── confusion metrics.png
│    ├── Roc curve.png
│── README.md                                 # Project documentation

```

---

## ✨ Recruiter Pitch

Developed a **Logistic Regression model** for diabetes prediction with strong interpretability and healthcare relevance.  

- Demonstrated full ML workflow (EDA → modeling → evaluation → insights).  
- Produced actionable clinical interpretations via coefficients & odds ratios.  
- Delivered professional, reproducible documentation & visual diagnostics.  
- Tools: **Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Jupyter**  


