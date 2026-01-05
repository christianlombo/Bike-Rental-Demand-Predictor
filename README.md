# Bike Rental Demand Predictor

## Project Overview
A bike rental system that uses regression techniques to predict demand and estimate the number of bikes required based on weather and seasonal data.

This project applies machine-learning regression to predict bike rental demand by analysing historical usage patterns alongside weather and seasonal variables. By modelling these factors, the system predicts a continuous numerical value representing the expected number of bike rentals, making regression the most suitable approach for this problem.

---

## Technology Stack & Dataset

### Tech Stack
* **Language:** Python 3.13
* **Core Library:** Scikit-Learn (RandomForestRegressor)
* **Data Processing:** Pandas 
* **Model Serialization:** Joblib

### The Dataset
* **Source:** [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data) (Kaggle).
* **Features:** Timestamp, Season, Holiday, Working Day, Weather, Temp, Humidity, Windspeed.
* **Target:** `count` (Total rentals per hour).

---

## How to Run
1. Training the Model
- Run the training script to process the raw timestamps and train the Random Forest Regressor.
- python train.py
* Output: Displays the R² Score and Mean Absolute Error (MAE), then saves the model to the Models/ directory.

2. Running the Predictor
- Launch the interactive forecasting tool. You will be asked for a date, time, and weather conditions.
- python predict.py
  
* Example Interaction:Enter Date: 2025-07-14 17:00 Prediction: 412 bikes needed (Context: Monday, Rush Hour)

## Methodology: 
1. With regression models, raw timestamps (e.g., 2025-01-05 00:00) are difficult to interpret directly. To address this, the datetime column is decomposed into meaningful numerical features such as hour, month, and day of the week, which the model can effectively process. This transformation enables the model to learn non-linear and cyclical patterns in bike rental demand, including peak usage periods and seasonal trends.

2. The model makes use of Random Forest Regression rather than Linear Regression. Linear Regression assumes a straight-line relationship between variables, which is unsuitable here because bike demand fluctuates significantly throughout the day, week, and month due to multiple factors. Random Forest is the ideal choice as it is an ensemble machine learning method. It combines predictions from multiple 'weak learners' (individual decision trees) to improve overall accuracy and reliability. By aggregating these trees, the model effectively captures complex, non-linear relationships in the data while reducing the risk of overfitting.

3. Seasons are categorical variables, not numerical magnitudes (e.g., 1, 2, 3, 4). To address this, a One-Hot Encoding scheme was applied within a ColumnTransformer. This technique transforms the data so that the model treats each category as a distinct feature rather than a single numerical value. This ensures the regression model interprets the categorical data correctly, avoiding the introduction of unintended ordinal relationships.

# Model Performance
- R² Score or the coefficient of determination is a evalution metric which is use in regression to measure how much variation there is to the output can be explaned by the model's input features.
- Range is between 0 (no variance) - 1.00 (Perfect prediction)
- The model had a coefficient of determination of 0.80 (The model explains 80% of the variance in the data).

- Mean Absolute Error (MAE) is a metric whcih measures the average size of prediction errors.
- Mean Absolute Error (MAE): 40-50 bikes.

- Insight: The model successfully learned that demand is highest during working/school commute hours (8 AM & 5 PM) on weekdays, but shifts to afternoon hours (12 PM - 3 PM) on weekends.

---
## Future Improvements
- Hyperparameter Tuning: Use GridSearchCV to optimize the number of trees in the forest.
- External Data: Integrate a real-time Weather API into predict.py so users don't have to manually enter the temperature.
