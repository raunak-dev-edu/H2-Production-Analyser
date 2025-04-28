"""
Model Fine-tuning Script for Hydrogen Production Prediction

This script improves the ML models by:
1. Creating synthetic data with more realistic variations
2. Using hyperparameter tuning to prevent overfitting
3. Training models that produce meaningful variations in predictions
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create directory paths if they don't exist
os.makedirs('./ml/models', exist_ok=True)
os.makedirs('./ml/data', exist_ok=True)

# Define realistic relationships for hydrogen production
def realistic_h2_from_temp(temp):
    """
    Temperature affects H2 production with peak around 800-850°C
    Lower temps reduce conversion efficiency
    Higher temps increase thermal losses
    """
    # Normalized bell curve centered at 825°C
    base = 1.5  # base production
    max_effect = 1.0  # peak additional production
    center = 825  # optimal temperature
    width = 150  # width of the peak
    
    bell_factor = max_effect * np.exp(-((temp - center) ** 2) / (2 * width ** 2))
    return round(base + bell_factor, 7)

def realistic_h2_from_pressure(pressure):
    """
    Pressure increases H2 production up to a point (diminishing returns)
    """
    # Logarithmic relationship with diminishing returns
    base = 1.5
    max_effect = 1.0
    factor = 0.3 * np.log(1 + pressure/5)  # logarithmic increase with pressure
    return round(base + max_effect * factor, 7)

def realistic_h2_from_temp_pressure(temp, pressure):
    """
    Combined effect with some interaction between variables
    """
    # Base from individual effects
    temp_effect = realistic_h2_from_temp(temp) - 1.5
    pressure_effect = realistic_h2_from_pressure(pressure) - 1.5
    
    # Add interaction effect
    base = 1.5
    interaction = 0.2 * (temp/1000) * (pressure/25)  # small interaction effect
    
    return round(base + temp_effect + pressure_effect + interaction, 7)

def realistic_h2_from_biogas(flow):
    """
    Biogas flow has nearly linear relationship with H2 production
    """
    base = 0.5
    factor = 2.5 * (flow / 100)  # linear scaling with flow
    return round(base + factor, 7)

# Create synthetic datasets with realistic variations
print("Generating synthetic datasets with realistic variations...")

# Temperature vs H2
temp_range = np.linspace(600, 1000, 100)
temp_h2_data = pd.DataFrame({
    'temp': temp_range,
    'h2_production': [realistic_h2_from_temp(t) for t in temp_range]
})
temp_h2_data.to_csv('./ml/data/temp_vs_h2.csv', index=False)
print(f"Created temperature dataset with range: {temp_h2_data['h2_production'].min():.4f} to {temp_h2_data['h2_production'].max():.4f} kg/h")

# Pressure vs H2
press_range = np.linspace(5, 25, 100)
press_h2_data = pd.DataFrame({
    'press': press_range,
    'h2_production': [realistic_h2_from_pressure(p) for p in press_range]
})
press_h2_data.to_csv('./ml/data/press_vs_h2.csv', index=False)
print(f"Created pressure dataset with range: {press_h2_data['h2_production'].min():.4f} to {press_h2_data['h2_production'].max():.4f} kg/h")

# Temperature + Pressure vs H2
temp_press_data = []
for temp in np.linspace(600, 1000, 20):
    for press in np.linspace(5, 25, 20):
        h2 = realistic_h2_from_temp_pressure(temp, press)
        temp_press_data.append([temp, press, h2])

temp_press_h2_data = pd.DataFrame(temp_press_data, columns=['temp', 'press', 'h2_production'])
temp_press_h2_data.to_csv('./ml/data/temp_press_vs_h2.csv', index=False)
print(f"Created temp-press dataset with range: {temp_press_h2_data['h2_production'].min():.4f} to {temp_press_h2_data['h2_production'].max():.4f} kg/h")

# Biogas flow vs H2
biogas_range = np.linspace(10, 100, 100)
biogas_h2_data = pd.DataFrame({
    'biogas_mass': biogas_range,
    'h2_production': [realistic_h2_from_biogas(b) for b in biogas_range]
})
biogas_h2_data.to_csv('./ml/data/biogas_mass_vs_h2.csv', index=False)
print(f"Created biogas dataset with range: {biogas_h2_data['h2_production'].min():.4f} to {biogas_h2_data['h2_production'].max():.4f} kg/h")

# Function to train model with hyperparameter tuning
def train_optimized_model(X, y, model_name):
    """Train a model with hyperparameter tuning to avoid overfitting"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [2, 3, 4],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2]
    }
    
    # Create grid search with cross-validation
    gbr = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=gbr,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        verbose=0,
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} - Best parameters: {grid_search.best_params_}")
    print(f"{model_name} - MSE: {mse:.8f}, R²: {r2:.6f}")
    
    # Plot feature importance if more than one feature
    if X.shape[1] > 1:
        importances = best_model.feature_importances_
        feature_names = X.columns
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(importances)), importances)
        plt.xticks(range(len(importances)), feature_names)
        plt.title(f'Feature Importances - {model_name}')
        plt.tight_layout()
        plt.savefig(f'./ml/{model_name}_feature_importance.png')
    
    # Save model
    model_path = f'./ml/models/{model_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Model saved to {model_path}")
    return best_model

# Train models with hyperparameter tuning
print("\nTraining models with hyperparameter tuning...\n")

# Temperature vs H2
print("\n=== Training Temperature vs H2 Model ===")
X_temp = temp_h2_data[['temp']]
y_temp = temp_h2_data['h2_production']
train_optimized_model(X_temp, y_temp, 'gbr_model_temp_vs_h2')

# Pressure vs H2
print("\n=== Training Pressure vs H2 Model ===")
X_press = press_h2_data[['press']]
y_press = press_h2_data['h2_production']
train_optimized_model(X_press, y_press, 'gbr_model_press_vs_h2')

# Temperature + Pressure vs H2
print("\n=== Training Temperature+Pressure vs H2 Model ===")
X_temp_press = temp_press_h2_data[['temp', 'press']]
y_temp_press = temp_press_h2_data['h2_production']
train_optimized_model(X_temp_press, y_temp_press, 'gbr_model_temp_press_vs_h2')

# Biogas flow vs H2
print("\n=== Training Biogas Flow vs H2 Model ===")
X_biogas = biogas_h2_data[['biogas_mass']]
y_biogas = biogas_h2_data['h2_production']
train_optimized_model(X_biogas, y_biogas, 'gbr_model_biogas_mass_vs_h2')

# Generate sample predictions to validate model behavior
print("\nValidating model predictions with sample inputs...\n")

# Load the trained models
temp_model = pickle.load(open('./ml/models/gbr_model_temp_vs_h2.pkl', 'rb'))
press_model = pickle.load(open('./ml/models/gbr_model_press_vs_h2.pkl', 'rb'))
temp_press_model = pickle.load(open('./ml/models/gbr_model_temp_press_vs_h2.pkl', 'rb'))
biogas_model = pickle.load(open('./ml/models/gbr_model_biogas_mass_vs_h2.pkl', 'rb'))

# Print sample predictions
print("Temperature model predictions:")
for temp in [600, 700, 800, 900, 1000]:
    pred = round(float(temp_model.predict([[temp]])[0]), 7)
    print(f"  Temperature {temp}°C → H2: {pred} kg/h")

print("\nPressure model predictions:")
for press in [5, 10, 15, 20, 25]:
    pred = round(float(press_model.predict([[press]])[0]), 7)
    print(f"  Pressure {press} bar → H2: {pred} kg/h")

print("\nTemp+Press model predictions:")
for temp, press in [(700, 10), (800, 15), (900, 20)]:
    pred = round(float(temp_press_model.predict([[temp, press]])[0]), 7)
    print(f"  Temp {temp}°C, Press {press} bar → H2: {pred} kg/h")

print("\nBiogas model predictions:")
for flow in [10, 25, 50, 75, 100]:
    pred = round(float(biogas_model.predict([[flow]])[0]), 7)
    print(f"  Biogas flow {flow} kg/h → H2: {pred} kg/h")

print("\nModel fine-tuning complete. The models now provide realistic variations in predictions based on input parameters.")
