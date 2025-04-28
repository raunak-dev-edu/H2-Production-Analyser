import pickle
import os
import pandas as pd

# Test just the temperature model for now
try:
    model_path = os.path.join('./ml/models', 'gbr_model_temp_vs_h2.pkl')
    model = pickle.load(open(model_path, 'rb'))
    
    print("\nTEMPERATURE MODEL TEST")
    print("=====================")
    # Test a range of temperatures
    results = []
    for temp in [600, 650, 700, 750, 800, 850, 900, 950, 1000]:
        h2 = round(float(model.predict([[temp]])[0]), 7)
        results.append([temp, h2])
        print(f"Temperature: {temp}°C → H₂: {h2} kg/h")
    
    # Check for variation
    df = pd.DataFrame(results, columns=['temperature', 'h2_production'])
    min_h2 = df['h2_production'].min()
    max_h2 = df['h2_production'].max()
    range_h2 = max_h2 - min_h2
    
    print(f"\nH₂ production range: {min_h2} to {max_h2} kg/h")
    print(f"Variation across temperature range: {range_h2} kg/h")
    
    if range_h2 > 0.1:
        print("\nSUCCESS: Model shows meaningful variation in predictions!")
    else:
        print("\nFAILURE: Model still lacks meaningful variation in predictions.")
    
except Exception as e:
    print(f"Error: {str(e)}")
