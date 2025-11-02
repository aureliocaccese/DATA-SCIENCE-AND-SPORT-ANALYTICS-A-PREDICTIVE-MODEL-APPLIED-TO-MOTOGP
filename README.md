# 🏍️ Data Science and Sport Analytics: A Predictive Model Applied to MotoGP

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Thesis%20Work-orange.svg)

## 📖 Project Overview
This project applies **Data Science** and **Machine Learning** to **MotoGP race analytics**, developing predictive models to estimate the **probability of podium finishes** based on rider performance, circuit features, and historical trends from **2002 to 2025**.

---

## 📊 Data Collection
Data were extracted from the **official MotoGP website** using **Selenium** for automated scraping and then processed into a structured dataset.

**Main dataset columns:**
Position, Points, Rider, Team, Time/Gap, Grand Prix, Year, Event, Rider_normalized,
Race Title, Conditions, Temperature, Track Conditions, Humidity, Ground Temp,
Avg_Position_Last3, Podiums_Last5, Points_Streak, circuit_file, file, length_km,
width_m, right corners, left corners, Race_Number_Season, Rider_Experience,
Points_numeric, Season_Points_So_Far, Championship_Position, Team_Avg_Position,
Difficult_Conditions, Rider_Circuit_Avg, Podio


**Temporal range:** 2002–2025  
**Target variable:** `Podio` (binary: 1 = podium finish, 0 = no podium)

---

## 🧠 Methodology
The analytical workflow follows standard **Data Science** and **Machine Learning** methodology:

1. **Data Cleaning & Preprocessing**  
   - Handling missing and inconsistent values  
   - Feature encoding and normalization  

2. **Exploratory Data Analysis (EDA)**  
   - Correlation analysis and visualization  
   - Track-level and rider-level statistics  

3. **Feature Engineering**  
   - Derived features like *Avg_Position_Last3*, *Points_Streak*, and *Rider_Experience*  

4. **Model Training & Evaluation**  
   - Train/test split, cross-validation, and hyperparameter tuning  
   - Evaluation using metrics: **Accuracy**, **Precision**, **Recall**, **F1**, **ROC-AUC**

---

## 🤖 Implemented Models
The following models were trained and compared:

| Model | Description | Notes |
|--------|--------------|-------|
| Logistic Regression | Baseline classification model | Fast, interpretable |
| Random Forest | Ensemble of decision trees | High accuracy, robust |
| Gradient Boosting (XGBoost) | Sequential ensemble | Best performance overall |
| K-Nearest Neighbors (KNN) | Distance-based classifier | Useful for exploratory comparison |
| Support Vector Machine (SVM) | Hyperplane optimization | Effective with kernel tuning |

✅ **Best performance:** *Random Forest* and *XGBoost*  
📈 Evaluated with **cross-validation** and **ROC-AUC ≈ 0.88**

---

## 🧩 Tools & Libraries
- **Python 3.10+**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**
- **Selenium** (for data extraction)
- **Jupyter Notebook** (for experimentation)

---

## 📂 Repository Structure

