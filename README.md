

#  Explainable Machine Learning for Early Diabetes Detection

This repository contains a **fully working, interactive web application** that predicts the risk of diabetes using a machine learning model and provides **explainable AI (XAI)** insights using SHAP (SHapley Additive exPlanations).

🔗 Live Demo: https://diabetes-xai-d9j38jmen65mn4hk3zksmu.streamlit.app/

---

##  Project Description

Early diabetes detection is crucial for preventive healthcare. In this project, we train a **Random Forest classifier** on clinical patient data to predict whether a person is at risk of diabetes. Unlike typical black-box models, our application also provides **transparent explanations** of the prediction using SHAP values, making it more interpretable for clinicians and users.

---

##  Key Features

-  **Interactive Web App** built with Streamlit  
-  **Diabetes Risk Prediction** using a Random Forest model  
-  **Explainability (SHAP)** to interpret feature impact  
-  Intuitive user interface for entering patient data  
-  Deployment via **Streamlit Cloud**  

---

##  Repository Structure

```

diabetes-xai/
│
├── app.py                 # Streamlit web app
├── model.py               # Model training
├── diabetes_model.pkl     # Saved trained model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── data/
└── diabetes.csv       # Dataset (PIMA Indians Diabetes)

````

---

##  Dataset

We used the **PIMA Indians Diabetes Dataset** from Kaggle:

 https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Features:

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Skinfold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes risk |
| Age | Age in years |
| Outcome | Target (0 = No Diabetes, 1 = Diabetes) |

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/diabetes-xai.git
cd diabetes-xai
````

---

### 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

### 3. Train the Model (Optional)

If the model file is not present, train the model:

```bash
python model.py
```

This will create the file:

```
diabetes_model.pkl
```

---

##  Running the App Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the link shown in the terminal (usually [http://localhost:8501](http://localhost:8501)) to interact with it.

---

##  How It Works

1. User enters clinical information such as glucose level, BMI, age, etc.
2. Model predicts whether the person has diabetes.
3. SHAP waterfall plot shows which features contributed most to the decision, giving transparency.

---

##  Explainability (SHAP)

The model uses **SHAP (SHapley Additive exPlanations)** to provide a global and local understanding of predictions.
SHAP outputs the contribution of each feature to the predicted risk for the current patient.

---

## Live Deployment

🎉 The app is deployed online using Streamlit Cloud:

 [https://diabetes-xai-d9j38jmen65mn4hk3zksmu.streamlit.app/](https://diabetes-xai-d9j38jmen65mn4hk3zksmu.streamlit.app/)

Users can interact with the model in real-time via the web.

---

## 🧾 Technologies Used

| Category       | Tools           |
| -------------- | --------------- |
| Language       | Python          |
| ML Library     | scikit-learn    |
| Explainability | SHAP            |
| Frontend UI    | Streamlit       |
| Deployment     | Streamlit Cloud |
| Data Handling  | pandas, numpy   |
| Visualization  | matplotlib      |

---

##  Sample Usage

1. Enter patient parameters:

   * Glucose
   * BMI
   * Blood pressure
   * Insulin, etc.
2. Click **Predict Diabetes**.
3. View prediction and SHAP explanation plot.

---

## Contributions

This repo is designed for academic projects. If you would like to extend it (e.g., add more models, visualization, deployment options), feel free to contribute!

---

## References

* Kaggle PIMA Diabetes Dataset
* SHAP documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
* Streamlit documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
```
