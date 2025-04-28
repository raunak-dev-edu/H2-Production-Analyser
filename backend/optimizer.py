import numpy as np
from sklearn.model_selection import ParameterGrid
from predictor import predict_h2, predict_lcoh

# Grid search to max H₂ or min LCOH
def optimize(objective='max_h2', constraints=None, resolution=5):
    if constraints is None:
        constraints = {}
    
    # Define parameter ranges with better resolution
    grid = {
        'temperature': np.linspace(500, 900, resolution),
        'pressure': np.linspace(1, 30, resolution),
        'biogas_flow': np.linspace(0.1, 1, resolution),
    }
    
    # Apply any constraints to narrow the search space
    for param, constraint in constraints.items():
        if param in grid and isinstance(constraint, dict):
            min_val = constraint.get('min')
            max_val = constraint.get('max')
            
            if min_val is not None and max_val is not None:
                grid[param] = np.linspace(min_val, max_val, resolution)
            elif min_val is not None:
                grid[param] = np.linspace(min_val, grid[param][-1], resolution)
            elif max_val is not None:
                grid[param] = np.linspace(grid[param][0], max_val, resolution)
    
    best = None
    results = []
    
    for params in ParameterGrid(grid):
        # Skip if any constraints are violated
        skip = False
        for param, constraint in constraints.items():
            if param not in params:
                continue
                
            if isinstance(constraint, dict):
                min_val = constraint.get('min')
                max_val = constraint.get('max')
                
                if (min_val is not None and params[param] < min_val) or \
                   (max_val is not None and params[param] > max_val):
                    skip = True
                    break
            elif isinstance(constraint, (int, float)) and params[param] != constraint:
                skip = True
                break
        
        if skip:
            continue
            
        # Calculate metrics
        h2_result = predict_h2(params)
        lcoh_result = predict_lcoh(params)
        
        h2_production = h2_result.get('h2_production', 0)
        lcoh_value = lcoh_result.get('value', float('inf'))
        
        # Calculate score based on objective
        if objective == 'max_h2':
            score = h2_production
        elif objective == 'min_lcoh':
            score = -lcoh_value
        elif objective == 'balanced':
            # Balanced objective: maximize H2 while minimizing LCOH
            # Normalize values to 0-1 range and combine
            norm_h2 = h2_production / 10  # Assuming max H2 is around 10 kg/h
            norm_lcoh = 1 - min(lcoh_value / 10, 1)  # Assuming max LCOH is around 10 $/kg
            score = 0.6 * norm_h2 + 0.4 * norm_lcoh  # Weighted combination
        else:
            score = h2_production  # Default to max_h2
            
        # Track result for this parameter set
        result = {
            'params': dict(params),
            'h2_production': h2_production,
            'lcoh': lcoh_value,
            'score': score
        }
        results.append(result)
        
        # Update best result
        if best is None or score > best['score']:
            best = result
    
    # Sort results by score in descending order
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Return top 3 results along with the best one
    return {
        'best': best,
        'top_results': sorted_results[:3] if len(sorted_results) >= 3 else sorted_results
    }