"""
Simple script to retrain models using the current scikit-learn version.
This ensures model compatibility with the deployment environment.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Function to train and save a model
def train_and_save_model(X, y, model_name):
    print(f"Training model: {model_name}")
    # Create a gradient boosting regressor
    model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    return model

# Train temperature vs H2 model
def train_temp_vs_h2():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "temp_vs_h2.csv"))
        X = df[['Temperature']].values
        y = df['H2_Production'].values
        return train_and_save_model(X, y, "gbr_model_temp_vs_h2")
    except Exception as e:
        print(f"Error training temp_vs_h2 model: {str(e)}")
        return None

# Train pressure vs H2 model
def train_press_vs_h2():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "press_vs_h2.csv"))
        X = df[['Pressure']].values
        y = df['H2_Production'].values
        return train_and_save_model(X, y, "gbr_model_press_vs_h2")
    except Exception as e:
        print(f"Error training press_vs_h2 model: {str(e)}")
        return None

# Train temperature and pressure vs H2 model
def train_temp_press_vs_h2():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "temp_press_vs_h2.csv"))
        X = df[['Temperature', 'Pressure']].values
        y = df['H2_Production'].values
        return train_and_save_model(X, y, "gbr_model_temp_press_vs_h2")
    except Exception as e:
        print(f"Error training temp_press_vs_h2 model: {str(e)}")
        return None

# Train biogas mass flow vs H2 model
def train_biogas_mass_vs_h2():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "biogas_mass_vs_h2.csv"))
        X = df[['Biogas_Mass_Flow']].values
        y = df['H2_Production'].values
        return train_and_save_model(X, y, "gbr_model_biogas_mass_vs_h2")
    except Exception as e:
        print(f"Error training biogas_mass_vs_h2 model: {str(e)}")
        return None

# Train temperature vs LCOH model
def train_temp_vs_lcoh():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "Temp_vs_LCOH.csv"))
        X = df[['Temperature']].values
        y = df['LCOH'].values
        return train_and_save_model(X, y, "gbr_model_temp_vs_lcoh")
    except Exception as e:
        print(f"Error training temp_vs_lcoh model: {str(e)}")
        return None

# Train pressure vs LCOH model
def train_press_vs_lcoh():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "Press_vs_LCOH.csv"))
        X = df[['Pressure']].values
        y = df['LCOH'].values
        return train_and_save_model(X, y, "gbr_model_press_vs_lcoh")
    except Exception as e:
        print(f"Error training press_vs_lcoh model: {str(e)}")
        return None

# Train electricity vs OPEX/LCOH model
def train_electricity_vs_opex_lcoh():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "Electricity_vs_OPEX_LCOH.csv"))
        X = df[['Electricity_Cost']].values
        y = df['OPEX_LCOH'].values
        return train_and_save_model(X, y, "gbr_model_electricity_vs_opex_lcoh")
    except Exception as e:
        print(f"Error training electricity_vs_opex_lcoh model: {str(e)}")
        return None

# Train LCOH vs year model
def train_lcoh_vs_year():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "LCOH_vs_Year.csv"))
        X = df[['Year']].values
        y = df['LCOH'].values
        return train_and_save_model(X, y, "gbr_model_lcoh_vs_year")
    except Exception as e:
        print(f"Error training lcoh_vs_year model: {str(e)}")
        return None

# Train CAPEX vs year model
def train_capex_vs_year():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "CapEx_vs_Year.csv"))
        X = df[['Year']].values
        y = df['CapEx'].values
        return train_and_save_model(X, y, "gbr_model_capex_vs_year")
    except Exception as e:
        print(f"Error training capex_vs_year model: {str(e)}")
        return None

# Main function to train all models
def train_all_models():
    print("Starting model training...")
    
    # Train all models
    train_temp_vs_h2()
    train_press_vs_h2()
    train_temp_press_vs_h2()
    train_biogas_mass_vs_h2()
    train_temp_vs_lcoh()
    train_press_vs_lcoh()
    train_electricity_vs_opex_lcoh()
    train_lcoh_vs_year()
    train_capex_vs_year()
    
    print("Model training complete!")

if __name__ == "__main__":
    train_all_models()
