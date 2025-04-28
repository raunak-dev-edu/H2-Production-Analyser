import pickle
import os

print("Testing the fine-tuned models for prediction variations:")
print("="*50)

# Load models
models_dir = './ml/models'

try:
    temp_model = pickle.load(open(os.path.join(models_dir, 'gbr_model_temp_vs_h2.pkl'), 'rb'))
    press_model = pickle.load(open(os.path.join(models_dir, 'gbr_model_press_vs_h2.pkl'), 'rb'))
    temp_press_model = pickle.load(open(os.path.join(models_dir, 'gbr_model_temp_press_vs_h2.pkl'), 'rb'))
    biogas_model = pickle.load(open(os.path.join(models_dir, 'gbr_model_biogas_mass_vs_h2.pkl'), 'rb'))
    
    # Temperature model predictions
    print("\nTemperature Model Predictions:")
    print("-" * 40)
    for temp in [600, 700, 800, 900, 1000]:
        h2 = round(float(temp_model.predict([[temp]])[0]), 7)
        print(f"Temperature: {temp}°C  →  H₂: {h2} kg/h")
    
    # Pressure model predictions
    print("\nPressure Model Predictions:")
    print("-" * 40)
    for press in [5, 10, 15, 20, 25]:
        h2 = round(float(press_model.predict([[press]])[0]), 7)
        print(f"Pressure: {press} bar  →  H₂: {h2} kg/h")
    
    # Temperature+Pressure model predictions
    print("\nTemperature+Pressure Model Predictions:")
    print("-" * 40)
    for temp, press in [(700, 10), (800, 15), (900, 20)]:
        h2 = round(float(temp_press_model.predict([[temp, press]])[0]), 7)
        print(f"Temp: {temp}°C, Press: {press} bar  →  H₂: {h2} kg/h")
    
    # Biogas model predictions
    print("\nBiogas Flow Model Predictions:")
    print("-" * 40)
    for flow in [10, 25, 50, 75, 100]:
        h2 = round(float(biogas_model.predict([[flow]])[0]), 7)
        print(f"Biogas Flow: {flow} kg/h  →  H₂: {h2} kg/h")

except Exception as e:
    print(f"Error testing models: {str(e)}")
