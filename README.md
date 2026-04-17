# 🚦 Traffic Volume Forecasting Project

## 📌 Overview

This project aims to predict traffic volume using machine learning techniques based on historical traffic and weather data.

The project follows the **CRISP-DM methodology**, covering all stages from business understanding to deployment.

Final outcome:
A **Tuned Random Forest Regressor** model capable of accurately predicting traffic volume.

---

## 📊 Dataset

The dataset includes traffic and environmental features such as:

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

The project is structured according to CRISP-DM phases:

1. **Business Understanding**

   * Defined project objectives and problem scope

2. **Data Understanding**

   * Explored dataset structure and distributions
   * Identified missing values and patterns

3. **Data Preparation**

   * Cleaned data
   * Handled missing values and duplicates
   * Engineered new features

4. **Modeling**

   * Trained multiple models:

     * Linear Regression
     * Ridge Regression
     * Random Forest
     * Gradient Boosting
   * Performed hyperparameter tuning

5. **Evaluation**

   * Compared models using MAE, RMSE, and R²
   * Selected the best-performing model

6. **Deployment**

   * Built an interactive app using Streamlit
   * Enabled real-time predictions

---

## 📁 Project Structure

```bash
traffic-volume-forecasting-project/
│
├── data/                 # raw and processed datasets
│   ├── raw/              # original dataset
│   └── processed/        # cleaned + feature engineered data
│
├── notebooks/            # CRISP-DM phase notebooks
│   ├── Phase_1_Business_Understanding.ipynb
│   ├── Phase_2_Data_Understanding.ipynb
│   ├── Phase_3_Data_Preparation.ipynb
│   ├── Phase_4_Modeling.ipynb
│   ├── Phase_5_Evaluation.ipynb
│   └── Phase_6_Deployment.ipynb
│
├── src/                  # reusable pipeline scripts
│   └── final_pipeline.py
│
├── models/               # saved trained model
│   └── tuned_random_forest_model.pkl
│
├── app.py                # Streamlit application
├── requirements.txt      # project dependencies
└── README.md             # project documentation
```

## 🤖 Models

The following models were trained and evaluated:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor

### ✅ Final Model

* **Tuned Random Forest Regressor**
* Achieved strong predictive performance on test data

---

## 📈 Results

* Best Model: Tuned Random Forest
* Test R² Score: ~0.966
* Low prediction error (MAE & RMSE)
* Model generalizes well to unseen data

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
* Generate real-time traffic predictions
* Visualize results interactively

---

## 👩‍💻 Author

**Zeina Abouelmagd**

---
