# Predictive Maintenance System for Industrial Equipment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predective-maintenance-d776wru8db4vgrkemj8jzp.streamlit.app/)

## Project Overview
This project implements a machine learning solution for predictive maintenance based on the AI4I 2020 Predictive Maintenance Dataset. The primary objective is to develop a robust classification model capable of predicting mechanical failures by analyzing real-time sensor data, including temperature, rotational speed, torque, and tool wear.

Live Application: [Predictive Maintenance Dashboard]([https://predective-maintenance-d776wru8db4vgrkemj8jzp.streamlit.app/](https://predective-maintenance-valentisecanell.streamlit.app/))

## Problem Statement
In industrial manufacturing, unexpected equipment failure results in substantial operational costs and production downtime. Maintenance strategies are typically categorized into two types:
1. **Reactive Maintenance:** Repairs are performed after a failure occurs, leading to high emergency costs.
2. **Preventive Maintenance:** Scheduled repairs that often lead to unnecessary interventions on functional equipment.

This project proposes a **Predictive Maintenance** approach, which utilizes sensor data to identify failure patterns, enabling interventions only when a significant risk is detected.

## Technical Methodology

### Data Management and Leakage Prevention
To ensure the model's reliability in a real-world scenario, rigorous data cleaning was performed. Features such as failure type categories (TWF, HDF, PWF, OSF, RNF) were removed, as they constitute data leakage—information that would not be available prior to the occurrence of a failure.

### Model Architecture and Training
* **Class Imbalance:** Given that failure events represent only 3.4% of the total observations, the Random Forest algorithm was configured with balanced class weights to improve sensitivity toward the minority class.
* **Pipeline Integration:** A standardized pipeline was implemented using `ColumnTransformer` to manage numerical scaling (StandardScaler) and categorical encoding (One-Hot Encoding).
* **Validation Strategy:** To guarantee performance stability, **Monte Carlo Cross-Validation** was executed with 100 iterations, providing a statistical distribution of the F1-Score.



## Results and Performance
* **F1-Score:** Approximately 0.85, demonstrating a balanced trade-off between Precision and Recall.
* **ROC-AUC:** 0.94, indicating high discriminative power.
* **Feature Importance:** Torque and Tool Wear were identified as the most significant predictors of equipment failure.

## Deployment
The final model is deployed as an interactive web application via **Streamlit**. The deployment utilizes the serialized model pipeline (`joblib`), ensuring that the preprocessing logic remains consistent between the training phase and real-time inference.

## Repository Structure
* `app.py`: Source code for the Streamlit web application.
* `Predictive_Maintenance.ipynb`: Comprehensive notebook containing data exploration, training, and validation.
* `predictive_maintenance_model.pkl`: Serialized model pipeline for production use.
* `requirements.txt`: Environment dependencies.

## Installation and Local Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
