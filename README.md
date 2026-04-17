# 🚦 Traffic Volume Forecasting Project

## 📌 Overview

This project predicts traffic volume using machine learning models based on historical traffic and weather data.

The project follows the **CRISP-DM methodology**:

* Business Understanding
* Data Understanding
* Data Preparation
* Modeling
* Evaluation
* Deployment

The final model is a **Tuned Random Forest Regressor**, achieving strong predictive performance.

---

## 📊 Dataset

The dataset includes:

* Date and time
* Weather conditions
* Temperature
* Holiday indicators
* Traffic volume (target variable)

Data is organized into:

* `data/raw/` → original dataset
* `data/processed/` → cleaned and feature-engineered data

---

## 🧠 Project Workflow

### 1. Business Understanding

* Defined objectives and project scope

### 2. Data Understanding

* Explored dataset structure
* Identified patterns, distributions, and missing values

### 3. Data Preparation

* Cleaned dataset
* Handled missing values and duplicates
* Engineered new features

### 4. Modeling

* Trained multiple models:

  * Linear Regression
  * Ridge Regression
  * Random Forest
  * Gradient Boosting
* Performed hyperparameter tuning

### 5. Evaluation

* Compared models using MAE, RMSE, and R²
* Selected best-performing model

### 6. Deployment

* Built an interactive Streamlit app
* Enabled real-time predictions

---

## 📁 Project Structure

```bash
traffic-volume-forecasting-project/
│
├── data/                         # datasets
│   ├── raw/                      # original dataset
│   └── processed/                # cleaned + engineered data
│
├── notebooks/                    # CRISP-DM notebooks
│   ├── Phase_1_Business_Understanding.ipynb
│   ├── Phase_2_Data_Understanding.ipynb
│   ├── Phase_3_Data_Preparation.ipynb
│   ├── Phase_4_Modeling.ipynb
│   ├── Phase_5_Evaluation.ipynb
│   └── Phase_6_Deployment.ipynb
│
├── src/                          # modular pipeline scripts
│   ├── cleaning.py               # data cleaning functions
│   ├── feature_engineering.py    # feature creation
│   ├── train_best_model.py       # model training
│   ├── evaluate.py               # model evaluation
│   ├── predict_best_model.py     # prediction logic
│   └── final_pipeline.py         # end-to-end pipeline
│
├── models/                       # saved trained model
│   └── tuned_random_forest_model.pkl
│
├── app.py                        # Streamlit application
├── requirements.txt              # dependencies
└── README.md                     # documentation
```

---

## 🤖 Models

The following models were trained and evaluated:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor

### ✅ Final Model

* **Tuned Random Forest Regressor**
* Best performance on test data

---

## 📈 Results

* Best Model: Tuned Random Forest
* Test R² Score: ~0.966
* Low prediction error (MAE & RMSE)
* Strong generalization to unseen data

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/traffic-volume-forecasting-project.git
cd traffic-volume-forecasting-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 🚀 Deployment

The model is deployed using **Streamlit**, allowing users to:

* Input feature values
* Generate real-time predictions
* Visualize results

---

## 👩‍💻 Author

**Zeina Abouelmagd**
