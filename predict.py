import pandas as pd
import joblib
import os

MODEL_PATH = 'Models/bike_demand_model.pkl'

def get_season(month):
    # 1=Spring, 2=Summer, 3=Fall, 4=Winter (Kaggle Dataset Mapping)
    if 3 <= month <= 5: return 1
    elif 6 <= month <= 8: return 2
    elif 9 <= month <= 11: return 3
    else: return 4

def main():
    print("--- BIKE DEMAND PREDICTOR ---")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train.py first!")
        return
    model = joblib.load(MODEL_PATH)
    
    while True:
        print("\nType 'q' or 'Q' to quit.")
        
        # 2. Get User Inputs
        date_input = input("Enter Date & Time (YYYY-MM-DD HH:00): ")
        if date_input.lower() == 'q': break
        
        try:
            temp_input = float(input("Enter Temperature (Celsius): "))
            weather_input = int(input("Choose weather option: \n1=Clear \n2=Mist \n3=Light Rain \n4=Heavy Rain \nOption:"))
        except ValueError:
            print("Invalid number. Try again.")
            continue
            
        # 3. Feature Engineering 
        try:
            dt = pd.to_datetime(date_input)
            
            # Create a dictionary with ALL the columns the model is expecting
            input_data = {
                'season': [get_season(dt.month)],
                'holiday': [0],          
                'workingday': [0 if dt.dayofweek >= 5 else 1], 
                'weather': [weather_input],
                'temp': [temp_input],
                'atemp': [temp_input],  
                'humidity': [60],       
                'windspeed': [12],       
                'hour': [dt.hour],
                'month': [dt.month],
                'day_of_week': [dt.dayofweek]
            }
            
            # Convert to DataFrame
            df_input = pd.DataFrame(input_data)
            
            # 4. Predict
            prediction = model.predict(df_input)[0]
            
            print(f"\n PREDICTION: {int(prediction)} bikes needed")
            print(f"   (Context: {dt.day_name()}, Hour {dt.hour}, Season {input_data['season'][0]})")
            
        except Exception as e:
            print(f"Error processing input: {e}")

if __name__ == "__main__":
    main()