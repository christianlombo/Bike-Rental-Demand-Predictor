import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = 'Data/train.csv'
MODEL_PATH = 'Models/bike_demand_model.pkl'

def feature_eng(df):
    """Extracting only the feature we will need from the raw datetime column"""
    #Converting datetime text to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek

    drop_cols = ['datetime', 'casual', 'registered']
    df = df.drop(columns = [c for c in drop_cols if c in df.columns], errors='ignore')

    return df

def train():
    print("Training started")

    if not os.path.exists(data):
        print(f"Error: {data} not found")
        return
    df = pd.read_csv(data)

    print("Engineering features")
    df = feature_eng(df)

    y = df['count']
    X = df.drop(columns=['count'])

    X_train, X_test,  y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    categorical_features = ['season', 'weather', 'day_of_week']

    preproc = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline([
        ('preprocessor', preproc),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("Training random forest")
    model_pipeline.fit(X_train, y_train)

    preditions = model_pipeline.predict(X_test)
    mean_abs_err = mean_absolute_error(y_test, preditions)
    r2 = r2_score(y_test, preditions)

    print(f"\nModel Performance:")
    print(f"R^2 Score: {r2:.2f} (1.0 is perfect)")
    print(f"Mean Absolute Error: {mean_abs_err:.1f} bikes")
    print(f"  -> On average, our prediction is off by about {int(mean_abs_err)} bikes.")
    
    # 8. Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()