from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

print(app.static_folder)


model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = {
            'Distance_km': float(request.form['distance']) if request.form['distance'] else None,
            'Preparation_Time_min': float(request.form['prep_time']) if request.form['prep_time'] else None,
            'Courier_Experience_yrs': float(request.form['experience']) if request.form['experience'] else None,
            'Weather': request.form['weather'],
            'Traffic_Level': request.form['traffic'],
            'Time_of_Day': request.form['time_of_day'],
            'Vehicle_Type': request.form['vehicle']
        }
        
        
        if any(value is None for value in data.values()):
            raise ValueError("Please ensure all fields are filled correctly.")
        
        
        input_df = pd.DataFrame([data])
        
        
        processed_input = preprocessor.transform(input_df)
        
    
        prediction = model.predict(processed_input)[0]
        
        return render_template('index.html', 
                            prediction_text=f'Estimated Delivery Time: {prediction:.1f} minutes',
                            model_used=f'Best Model: {type(model).__name__}')

    except Exception as e:
        return render_template('index.html', 
                            prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
