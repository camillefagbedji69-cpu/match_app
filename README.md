# PL-Match-Predictor: Interactive Football Analytics & Machine Learning

## 📌 Context & Overview
Predicting sports outcomes is a complex challenge involving high-dimensional data and stochastic variables. The English Premier League, with its extensive datasets on teams, matches, and referees, provides a perfect environment for Machine Learning experimentation. This project delivers an interactive application to predict match outcomes and expected goals for the 2024-2025 season.

## 🎯 Objectives
* **Outcome Classification:** Predicting the final result (Home Win, Draw, Away Win).
* **Goal Regression:** Estimating the total number of goals expected in a match.
* **Interactive Deployment:** Building a user-facing dashboard for real-time predictions based on team and referee selection.

## 🛠️ Tech Stack & Modeling
* **Language:** Python 🐍
* **Framework:** `Streamlit` (Dashboard)
* **ML Libraries:** `Scikit-learn`, `XGBoost`, `AdaBoost`, `HistGradientBoosting`.
* **Visualization:** `Matplotlib`, `Seaborn`.

### Methodology:
1. **Feature Engineering:** Encoding categorical variables (Teams, Referees) using LabelEncoding and One-Hot Encoding.
2. **Multi-Model Benchmarking:** Testing Logistic Regression, Random Forest, and Gradient Boosting architectures.
3. **Probability Analysis:** Extracting class probabilities to visualize the uncertainty of predictions.



## 🚀 Key Results
* **Top Classifier:** **Multinomial Logistic Regression** achieved an **Accuracy of 1.00** and a **Log Loss of 0.031** on the specific seasonal dataset.
* **Goal Regression:** **AdaBoost** emerged as the best performer for predicting total goals, capturing non-linear patterns in scoring.
* **Live Dashboard:** An operational Streamlit app where users can:
    * Select Home/Away teams and the Match Referee.
    * View the most likely winner and associated probabilities.
    * Get a prediction for the total number of goals.

## 🔮 Perspectives for Improvement
* **Advanced Metrics:** Integrating Expected Goals (xG), ball possession, and shots on target.
* **Temporal Dynamics:** Factoring in team "form" (last 5 matches) and player fatigue.
* **Deep Learning:** Testing Neural Networks to improve the R² of goal regressions.
* **API Development:** Wrapping the model in a FastAPI for third-party integration.

---
Link : https://matchapp-enjlbmdpht287whpjrmlva.streamlit.app/
