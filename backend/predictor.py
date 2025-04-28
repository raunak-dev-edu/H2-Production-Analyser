import pickle, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import os

# Updated path to models directory
MODEL_DIR = "../ml/models"

# Load model with proper error handling
def load_model(name):
    try:
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found")
            return None
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        print(f"Error loading model {name}: {str(e)}")
        return None

# Load all GBR models from the new ml/models directory
models = {
    'temp_h2':       load_model('gbr_model_temp_vs_h2'),
    'press_h2':      load_model('gbr_model_press_vs_h2'),
    'temp_press_h2': load_model('gbr_model_temp_press_vs_h2'),
    'flow_h2':       load_model('gbr_model_biogas_mass_vs_h2'),
    'temp_lcoh':     load_model('gbr_model_temp_vs_lcoh'),
    'press_lcoh':    load_model('gbr_model_press_vs_lcoh'),
    'elec_lcoh':     load_model('gbr_model_electricity_vs_opex_lcoh'),
    'year_lcoh':     load_model('gbr_model_lcoh_vs_year'),
    'year_capex':    load_model('gbr_model_capex_vs_year'),
}

# Create sample trend data frame for visualization
def create_trend_data():
    """Create synthetic trend data for various variables"""
    data = {}
    
    # Temperature trends (20-100°C)
    temp_range = range(20, 101, 10)
    data['temperature'] = pd.DataFrame({
        'temperature': temp_range,
        'H2_output': [0.8 + 0.08*t for t in temp_range],  # H2 increases with temperature
        'LCOH': [4.5 - 0.02*t for t in temp_range],  # LCOH decreases with temperature
        'CapEx': [100000 + 1000*t for t in temp_range]  # CapEx increases with temperature
    })
    
    # Pressure trends (1-30 bar)
    pressure_range = range(1, 31, 3)
    data['pressure'] = pd.DataFrame({
        'pressure': pressure_range,
        'H2_output': [0.5 + 0.15*p for p in pressure_range],  # H2 increases with pressure
        'LCOH': [5.0 - 0.1*p for p in pressure_range],  # LCOH decreases with pressure
        'CapEx': [80000 + 2000*p for p in pressure_range]  # CapEx increases with pressure
    })
    
    # Biogas flow trends (10-100 kg/h)
    flow_range = range(10, 101, 10)
    data['biogas_flow'] = pd.DataFrame({
        'biogas_flow': flow_range,
        'H2_output': [0.1*f for f in flow_range],  # H2 increases linearly with flow
        'LCOH': [6.0 - 0.03*f for f in flow_range],  # LCOH decreases with flow
        'CapEx': [50000 + 500*f for f in flow_range]  # CapEx increases with flow
    })
    
    # Electricity price trends ($0.05-0.25/kWh)
    price_points = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    data['electricity_price'] = pd.DataFrame({
        'electricity_price': price_points,
        'H2_output': [5.0 for _ in price_points],  # H2 is constant with electricity price
        'LCOH': [2.0 + 10*p for p in price_points],  # LCOH increases with electricity price
        'CapEx': [100000 for _ in price_points]  # CapEx is constant with electricity price
    })
    
    # Year trends (2023-2030)
    year_range = range(2023, 2031)
    data['year'] = pd.DataFrame({
        'year': year_range,
        'H2_output': [5.0 + 0.5*(y-2023) for y in year_range],  # H2 increases over time (tech improvement)
        'LCOH': [4.0 - 0.3*(y-2023) for y in year_range],  # LCOH decreases over time
        'CapEx': [100000 - 5000*(y-2023) for y in year_range]  # CapEx decreases over time
    })
    
    return data

# Global trend data storage
trend_data = None

# Improved predict H₂ production function with better error handling and dynamic model selection
def predict_h2(p):
    try:
        if 'temperature' in p and 'pressure' in p:
            model_key = 'temp_press_h2'
            X = [[p['temperature'], p['pressure']]]
        elif 'temperature' in p:
            model_key = 'temp_h2'
            X = [[p['temperature']]]
        elif 'pressure' in p:
            model_key = 'press_h2'
            X = [[p['pressure']]]
        elif 'biogas_flow' in p:
            model_key = 'flow_h2'
            X = [[p['biogas_flow']]]
        else:
            return {'error': 'Missing required parameters. Please provide temperature, pressure, or biogas_flow.'}
        
        model = models[model_key]
        if model is None:
            return {'error': f'Model {model_key} not available', 'h2_production': 0.0}
        
        # Return the value with 7 decimal places
        prediction = round(float(model.predict(X)[0]), 7)
        return {'h2_production': prediction, 'model_used': model_key}
    except Exception as e:
        print(f"Error in predict_h2: {str(e)}")
        return {'error': str(e), 'h2_production': 0.0}

# Improved LCOH or CapEx prediction with better error handling
def predict_lcoh(p):
    try:
        if 'temperature' in p:
            model_key = 'temp_lcoh'
            X = [[p['temperature']]]
        elif 'pressure' in p:
            model_key = 'press_lcoh'
            X = [[p['pressure']]]
        elif 'electricity_price' in p:
            model_key = 'elec_lcoh'
            X = [[p['electricity_price']]]
        elif 'year' in p:
            if p.get('target', 'lcoh') == 'lcoh':
                model_key = 'year_lcoh'
            else:
                model_key = 'year_capex'
            X = [[p['year']]]
        else:
            return {'error': 'Missing required parameters. Please provide temperature, pressure, electricity_price, or year.'}
        
        model = models[model_key]
        if model is None:
            return {'error': f'Model {model_key} not available', 'value': 0.0}
        
        # Return the value with 7 decimal places
        prediction = round(float(model.predict(X)[0]), 7)
        return {'value': prediction}
    except Exception as e:
        print(f"Error in predict_lcoh: {str(e)}")
        return {'error': str(e), 'value': 0.0}

# Plot trend of H₂ vs variable with improved error handling
def plot_trend(var, chart_type=None):
    try:
        # Lazy load trend data on first call
        global trend_data
        if trend_data is None:
            trend_data = create_trend_data()
        
        # Make sure we have data for the requested variable
        if var not in trend_data:
            return {"error": f"No trend data for variable: {var}"}
        
        # Get the appropriate dataframe for this variable
        trend_df = trend_data[var]
        
        # Create a new figure with Agg backend
        plt.figure(figsize=(10, 6))
        
        # Map variable names to more readable labels
        var_labels = {
            'temperature': 'Temperature (°C)',
            'pressure': 'Pressure (bar)',
            'biogas_flow': 'Biogas Flow (kg/h)',
            'electricity_price': 'Electricity Price ($/kWh)',
            'year': 'Year'
        }
        
        # Map output variables to more readable labels
        output_labels = {
            'H2_output': 'H₂ Production (kg/h)',
            'LCOH': 'Levelized Cost of Hydrogen ($/kg)',
            'CapEx': 'Capital Expenditure ($)'
        }
        
        # Determine which output variable to plot based on chart_type parameter
        output_var = 'H2_output'  # Default
        
        # Override based on chart_type if specified
        if chart_type:
            if chart_type.lower() == 'lcoh':
                output_var = 'LCOH'
            elif chart_type.lower() == 'capex':
                output_var = 'CapEx'
            elif chart_type.lower() == 'h2_output':
                output_var = 'H2_output'
        # If no chart_type but specific variables, use smart defaults
        elif var in ['electricity_price', 'year']:
            output_var = 'LCOH'
        
        # Print debug info
        print(f"Generating chart: {var} vs {output_var}, chart_type={chart_type}")
        
        # Apply styling to the plot
        plt.plot(trend_df[var], trend_df[output_var], marker='o', linestyle='-', 
                linewidth=2, markersize=8, color='#1f77b4')
        
        # Add grid and styling
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel(var_labels.get(var, var), fontsize=12)
        plt.ylabel(output_labels.get(output_var, output_var), fontsize=12)
        plt.title(f'Effect of {var_labels.get(var, var)} on {output_labels.get(output_var, output_var)}', fontsize=14)
        
        # Add data points labels
        for i, (x, y) in enumerate(zip(trend_df[var], trend_df[output_var])):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode()
        plt.close('all')  # Ensure all figures are closed
        
        explanation = f"This chart shows how {var_labels.get(var, var)} affects {output_labels.get(output_var, output_var)} in hydrogen production systems. "
        if output_var == 'H2_output':
            explanation += "Higher values generally indicate better production efficiency."
        elif output_var == 'LCOH':
            explanation += "Lower LCOH values indicate more cost-effective hydrogen production."
        else:  # CapEx
            explanation += "This economic indicator helps evaluate the capital investment required for the process."
        
        # Return both the image and data for the frontend
        return {
            "image": f"data:image/png;base64,{img}",
            "explanation": explanation,
            "data": trend_df.to_dict(orient='records')
        }
    except Exception as e:
        print(f"Error in plot_trend: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Generate multi-parameter comparison chart with improved error handling
def compare_parameters(params_list, target='h2'):
    try:
        plt.figure(figsize=(12, 7))
        
        labels = []
        values = []
        
        for params in params_list:
            label_parts = []
            for key, value in params.items():
                if key in ['temperature', 'pressure', 'biogas_flow', 'electricity_price']:
                    label_parts.append(f"{key[:4]}:{value}")
            
            labels.append('\n'.join(label_parts))
            
            if target == 'h2':
                result = predict_h2(params)
                values.append(result['h2_production'])
            elif target == 'lcoh':
                result = predict_lcoh(params)
                values.append(result.get('value', 0))
        
        # Create bar chart
        plt.bar(range(len(values)), values, color='skyblue')
        plt.xticks(range(len(values)), labels, rotation=0)
        
        # Add value labels on top of each bar
        for i, v in enumerate(values):
            plt.text(i, v + 0.1, f"{v:.7f}", ha='center', fontsize=10)
        
        title = 'H₂ Production Comparison (kg/h)' if target == 'h2' else 'LCOH Comparison ($/kg)'
        plt.title(title, fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        # Prepare data table
        table_data = []
        for i, params in enumerate(params_list):
            row = {key: value for key, value in params.items()}
            row['result'] = values[i]
            table_data.append(row)
        
        return {
            "image": f"data:image/png;base64,{img}",
            "table": table_data
        }
    except Exception as e:
        print(f"Error in compare_parameters: {str(e)}")
        return {"error": str(e)}

# Create correlation heatmap for sensitivity analysis with improved error handling
def sensitivity_analysis():
    try:
        # Generate data for sensitivity analysis
        temp_range = np.linspace(600, 1000, 5)
        press_range = np.linspace(5, 25, 5)
        
        # Create a grid of temperature and pressure combinations
        temp_grid, press_grid = np.meshgrid(temp_range, press_range)
        
        # Calculate H2 production for each combination
        h2_values = np.zeros_like(temp_grid)
        for i in range(len(temp_range)):
            for j in range(len(press_range)):
                result = predict_h2({'temperature': temp_grid[j, i], 'pressure': press_grid[j, i]})
                h2_values[j, i] = result['h2_production']
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(h2_values, cmap='viridis', aspect='auto', 
                   extent=[temp_range[0], temp_range[-1], press_range[0], press_range[-1]])
        
        # Add colorbar and labels
        plt.colorbar(label='H₂ Production (kg/h)')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Pressure (bar)')
        plt.title('Sensitivity Analysis: H₂ Production vs Temperature and Pressure')
        
        # Add grid lines
        plt.grid(False)
        
        # Add text annotations
        for i in range(len(temp_range)):
            for j in range(len(press_range)):
                plt.text(temp_range[i], press_range[j], f'{h2_values[j, i]:.7f}', 
                         ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        # Calculate percentage changes for a baseline case
        baseline = predict_h2({'temperature': 800, 'pressure': 15})['h2_production']
        
        temp_sensitivity = []
        for t in temp_range:
            result = predict_h2({'temperature': t, 'pressure': 15})['h2_production']
            pct_change = ((result - baseline) / baseline) * 100
            temp_sensitivity.append({'temperature': t, 'h2': result, 'pct_change': pct_change})
        
        press_sensitivity = []
        for p in press_range:
            result = predict_h2({'temperature': 800, 'pressure': p})['h2_production']
            pct_change = ((result - baseline) / baseline) * 100
            press_sensitivity.append({'pressure': p, 'h2': result, 'pct_change': pct_change})
        
        return {
            "image": f"data:image/png;base64,{img}",
            "temperature_sensitivity": temp_sensitivity,
            "pressure_sensitivity": press_sensitivity,
            "baseline": baseline
        }
    except Exception as e:
        print(f"Error in sensitivity_analysis: {str(e)}")
        return {"error": str(e)}