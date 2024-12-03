# Calories Burnt Prediction

## Overview

This project aims to predict the number of calories burnt during physical activity based on various physiological and exercise parameters. By leveraging machine learning techniques, specifically **XGBoost** with **GridSearchCV**, the model is optimized for better performance.  

## Dataset

The dataset consists of the following columns:

| Column Name  | Description                                       |
|--------------|---------------------------------------------------|
| **Age**      | Age of the individual in years                   |
| **Height**   | Height of the individual in centimeters          |
| **Weight**   | Weight of the individual in kilograms            |
| **Duration** | Duration of the activity in minutes              |
| **Heart_Rate** | Average heart rate during the activity (bpm)   |
| **Body_Temp** | Body temperature during the activity (°C)       |
| **Calories** | Total calories burnt (target variable)           |

## Objective

To develop a machine learning model that accurately predicts the number of calories burnt based on input features.

## Approach

1. **Exploratory Data Analysis (EDA):**
   - Data cleaning and visualization.
   - Checking for missing values, outliers, and feature relationships.

2. **Feature Engineering:**
   - Normalization/Standardization of input features.
   - Feature importance analysis to identify the most significant predictors.

3. **Model Selection:**
   - Using **XGBoost Regressor** as the base model due to its efficiency and performance.

4. **Hyperparameter Tuning:**
   - Employing **GridSearchCV** to find the best combination of hyperparameters for the XGBoost model.

5. **Evaluation:**
   - Model evaluation using metrics like R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE).

## Model Development

### XGBoost with GridSearchCV

The following steps were performed to optimize the XGBoost model:

1. **Parameter Grid Definition:**

   ```python
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [3, 5, 7],
       'learning_rate': [0.01, 0.1, 0.2],
       'subsample': [0.8, 1.0],
       'colsample_bytree': [0.8, 1.0]
   }
   ```

2. **GridSearchCV Setup:**

   ```python
   from sklearn.model_selection import GridSearchCV
   from xgboost import XGBRegressor

   grid_search = GridSearchCV(
       estimator=XGBRegressor(),
       param_grid=param_grid,
       scoring='r2',
       cv=5,
       verbose=1,
       n_jobs=-1
   )
   ```

3. **Model Training and Best Parameters:**

   The GridSearchCV is fitted on the training data, and the best parameters are used to retrain the model for predictions.

4. **Model Evaluation:**

   Evaluation metrics for the final model include:

   - R² Score
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Mean Absolute Percentage Error (MAPE)

## Results

The model achieved the following performance metrics:

| Metric        | Value  |
|---------------|--------|
| **R² Score**  | *e.g., 0.85* |
| **MAE**       | *e.g., 10.5* |
| **MSE**       | *e.g., 125.3* |
| **MAPE**      | *e.g., 12.8%* |

## Requirements

To run the project, ensure the following libraries are installed:

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

Install the requirements using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/mukesh1996-ds/calories-burnt-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd calories-burnt-prediction
   ```

3. Run the model script:

   ```bash
   python calories_prediction.py
   ```

4. View results and metrics.

## File Structure

```
├── data/
│   ├── calories_data.csv      # Dataset
├── models/
│   ├── best_xgboost_model.pkl # Trained XGBoost model
├── scripts/
│   ├── calories_prediction.py # Main model script
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
```

## Future Work

- Enhance the feature set by adding more physiological and activity parameters.
- Implement other regression models for comparison.
- Explore deep learning techniques for further improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
