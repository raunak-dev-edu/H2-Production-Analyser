import pandas as pd
import numpy as np
import os

# Make sure the data directory exists
os.makedirs('data', exist_ok=True)

# Create sample data for LCOH vs Temperature
temp_range = np.linspace(500, 900, 9)
h2_output = np.array([3.2, 4.1, 5.0, 5.8, 6.3, 6.7, 6.9, 7.0, 6.8])
lcoh_values = np.array([5.2, 4.8, 4.5, 4.3, 4.2, 4.3, 4.5, 4.8, 5.1])

# Create DataFrame
lcoh_temp_df = pd.DataFrame({
    'temperature': temp_range,
    'H2_output': h2_output,
    'LCOH': lcoh_values
})

# Create similar data for pressure
press_range = np.linspace(1, 30, 7)
h2_output_press = np.array([4.1, 5.2, 5.9, 6.3, 6.5, 6.6, 6.5])
lcoh_values_press = np.array([4.9, 4.6, 4.4, 4.3, 4.4, 4.6, 4.9])

# Create DataFrame for pressure
lcoh_press_df = pd.DataFrame({
    'pressure': press_range,
    'H2_output': h2_output_press,
    'LCOH': lcoh_values_press
})

# Create sample data for year projections
years = np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
capex_values = np.array([12000000, 11500000, 11000000, 10500000, 10000000, 9800000, 9600000, 9500000])
lcoh_year_values = np.array([5.2, 5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.9])

# Create DataFrames
year_capex_df = pd.DataFrame({
    'year': years,
    'CapEx': capex_values
})

year_lcoh_df = pd.DataFrame({
    'year': years,
    'LCOH': lcoh_year_values
})

# Create a biogas flow dataset
biogas_flow = np.linspace(0.1, 1.0, 7)
h2_biogas = np.array([1.5, 3.0, 4.5, 5.5, 6.3, 6.8, 7.0])

biogas_df = pd.DataFrame({
    'biogas_flow': biogas_flow,
    'H2_output': h2_biogas
})

# Create sample data for electricity price vs OPEX and LCOH
elec_prices = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20])
opex_values = np.array([500000, 600000, 700000, 800000, 900000, 1000000, 1100000])
lcoh_elec_values = np.array([3.5, 4.0, 4.3, 4.6, 5.0, 5.5, 6.0])

elec_df = pd.DataFrame({
    'electricity_price': elec_prices,
    'OPEX': opex_values,
    'LCOH': lcoh_elec_values
})

# Use absolute path to ensure file is saved in the correct location
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, 'data', 'CapEx.xlsx')

# Create a combined Excel file with multiple sheets
with pd.ExcelWriter(excel_path) as writer:
    lcoh_temp_df.to_excel(writer, sheet_name='LCOH_vs_Temp', index=False)
    lcoh_press_df.to_excel(writer, sheet_name='LCOH_vs_Press', index=False)
    year_capex_df.to_excel(writer, sheet_name='CapEx_vs_Year', index=False)
    year_lcoh_df.to_excel(writer, sheet_name='LCOH_vs_Year', index=False)
    biogas_df.to_excel(writer, sheet_name='Biogas_vs_H2', index=False)
    elec_df.to_excel(writer, sheet_name='Elec_vs_LCOH', index=False)

print(f"Sample data created successfully at {excel_path}")
