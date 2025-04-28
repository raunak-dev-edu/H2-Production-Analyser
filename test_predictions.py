import pickle, os
import numpy as np
import pandas as pd

# Test temperature model
model_path = os.path.join('./ml/models', 'gbr_model_temp_vs_h2.pkl')
model = pickle.load(open(model_path, 'rb'))
print("TEMPERATURE MODEL PREDICTIONS:")
temps = [600, 700, 800, 900, 950]
for temp in temps:
    print(f"Temperature {temp}°C: H2 Production = {model.predict([[temp]])[0]}")

print("\n" + "="*50 + "\n")

# Test pressure model
model_path = os.path.join('./ml/models', 'gbr_model_press_vs_h2.pkl')
model = pickle.load(open(model_path, 'rb'))
print("PRESSURE MODEL PREDICTIONS:")
pressures = [5, 10, 15, 20, 25]
for press in pressures:
    print(f"Pressure {press} bar: H2 Production = {model.predict([[press]])[0]}")

print("\n" + "="*50 + "\n")

# Test temp-press model
model_path = os.path.join('./ml/models', 'gbr_model_temp_press_vs_h2.pkl')
model = pickle.load(open(model_path, 'rb'))
print("TEMPERATURE-PRESSURE MODEL PREDICTIONS:")
for temp in [700, 800, 900]:
    for press in [10, 15, 20]:
        print(f"Temp {temp}°C, Press {press} bar: H2 Production = {model.predict([[temp, press]])[0]}")

print("\n" + "="*50 + "\n")

# Check data files for variation
print("DATA FILE EXAMINATION:")
temp_df = pd.read_csv("./ml/data/temp_vs_h2.csv")
print(f"Temperature data: Min H2 = {temp_df['h2_production'].min()}, Max H2 = {temp_df['h2_production'].max()}")
print(f"Temperature range: {temp_df['temp'].min()} to {temp_df['temp'].max()}")
print(f"Variance in H2 production: {temp_df['h2_production'].var()}")

press_df = pd.read_csv("./ml/data/press_vs_h2.csv")
print(f"Pressure data: Min H2 = {press_df['h2_production'].min()}, Max H2 = {press_df['h2_production'].max()}")
print(f"Pressure range: {press_df['press'].min()} to {press_df['press'].max()}")
print(f"Variance in H2 production: {press_df['h2_production'].var()}")

temp_press_df = pd.read_csv("./ml/data/temp_press_vs_h2.csv")
print(f"Temp-Press data: Min H2 = {temp_press_df['h2_production'].min()}, Max H2 = {temp_press_df['h2_production'].max()}")
print(f"Temperature range: {temp_press_df['temp'].min()} to {temp_press_df['temp'].max()}")
print(f"Pressure range: {temp_press_df['press'].min()} to {temp_press_df['press'].max()}")
print(f"Variance in H2 production: {temp_press_df['h2_production'].var()}")
