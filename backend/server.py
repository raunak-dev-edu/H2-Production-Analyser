from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from predictor import predict_h2, predict_lcoh, plot_trend, compare_parameters, sensitivity_analysis
from optimizer import optimize
from emissions import calculate_emissions
from storage import save_session, get_sessions
import uvicorn
import numpy as np
import matplotlib
# Force matplotlib to use 'Agg' backend to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = FastAPI(title="H₂ Production Analyzer API", 
             description="API for hydrogen production prediction and economic analysis",
             version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://hydrogen-production-analyzer.windsurf.build"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_h2")
async def api_h2(p: dict):
    return predict_h2(p)

@app.post("/predict_lcoh")
async def api_lcoh(p: dict):
    return predict_lcoh(p)

@app.get("/plot_trend")
def api_plot(var: str = Query(default="temperature"), chart_type: str = Query(default=None)):
    """Generate trend charts for how various parameters affect H2 production and economics."""
    print(f"Plot trend requested for var={var}, chart_type={chart_type}")
    
    try:
        # Pass chart_type to the predictor function
        result = plot_trend(var, chart_type)
        return result
    except Exception as e:
        print(f"Error in /plot_trend: {str(e)}")
        return {"error": str(e)}

@app.post("/optimize")
def api_opt(p: dict):
    return optimize(p.get("objective", "max_h2"), 
                   p.get("constraints", {}),
                   p.get("resolution", 5))

@app.post("/emissions")
def api_em(p: dict):
    return calculate_emissions(**p)

@app.post("/simulate")
def api_sim(p: dict):
    res = []
    for s in p["scenarios"]:
        h2_result = predict_h2(s)
        lcoh_result = predict_lcoh(s)
        emissions_result = calculate_emissions(
            s.get("ch4_percent", 0),
            s.get("co2_percent", 0),
            h2_result["h2_production"]
        )
        combined_result = {**s, **h2_result, **lcoh_result, **emissions_result}
        res.append(combined_result)
    return {"results": res}

@app.post("/whatif")
async def api_whatif(data: dict):
    print(f"Running what-if analysis with data: {data}")
    # Extract parameters from the request
    param = data.get('param', 'temperature')
    from_val = data.get('from', 700)
    to_val = data.get('to', 800)
    
    # Generate sample data for chart
    if param == 'temperature':
        x_range = np.linspace(from_val, to_val, 10)
        y_values = [predict_h2({'temperature': x})['h2_production'] for x in x_range]
        x_label = 'Temperature (°C)'
        y_label = 'H₂ Production (kg/h)'
    elif param == 'pressure':
        x_range = np.linspace(from_val, to_val, 10)
        y_values = [predict_h2({'pressure': x})['h2_production'] for x in x_range]
        x_label = 'Pressure (bar)'
        y_label = 'H₂ Production (kg/h)'
    elif param == 'biogas_flow':
        x_range = np.linspace(from_val, to_val, 10)
        y_values = [predict_h2({'biogas_flow': x})['h2_production'] for x in x_range]
        x_label = 'Biogas Flow (kg/h)'
        y_label = 'H₂ Production (kg/h)'
    elif param == 'electricity_price':
        x_range = np.linspace(from_val, to_val, 10)
        y_values = [predict_lcoh({'electricity_price': x}).get('value', 0) for x in x_range]
        x_label = 'Electricity Price ($/kWh)'
        y_label = 'LCOH ($/kg)'
    else:
        return {"error": f"Unknown parameter: {param}"}
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_values, 'o-', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'What-If Analysis: {x_label} vs {y_label}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data point labels
    for i, (x, y) in enumerate(zip(x_range, y_values)):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Convert plot to base64 image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    explanation = f"When {param} changes from {from_val} to {to_val}, "
    if param == 'temperature' or param == 'pressure' or param == 'biogas_flow':
        explanation += f"hydrogen production increases from {y_values[0]:.4f} kg/h to {y_values[-1]:.4f} kg/h."
    else:
        explanation += f"the levelized cost of hydrogen changes from ${y_values[0]:.4f}/kg to ${y_values[-1]:.4f}/kg."
    
    return {
        "image": f"data:image/png;base64,{img}",
        "explanation": explanation,
        "data": {
            "x": x_range.tolist(),
            "y": y_values,
            "x_label": x_label,
            "y_label": y_label
        }
    }

@app.post("/history/save")
def api_save(p: dict):
    save_session(p["session"])
    return {"status": "ok"}

@app.get("/history")
def api_hist():
    return {"sessions": get_sessions()}

# New endpoints
@app.post("/compare")
def api_compare(p: dict):
    """Compare multiple parameter combinations and visualize the results"""
    return compare_parameters(
        p.get("parameter_sets", []),
        p.get("target", "h2")
    )

@app.get("/sensitivity")
def api_sensitivity():
    """Perform sensitivity analysis on input parameters"""
    return sensitivity_analysis()

@app.post("/integrated_analysis")
def api_integrated(p: dict):
    """Perform comprehensive analysis with multiple metrics"""
    # Get hydrogen production
    h2_result = predict_h2(p)
    
    # Get economic metrics
    lcoh_result = predict_lcoh(p)
    
    # Get emissions data if CH4 and CO2 percentages provided
    emissions_result = {}
    if "ch4_percent" in p and "co2_percent" in p:
        emissions_result = calculate_emissions(
            p["ch4_percent"],
            p["co2_percent"],
            h2_result["h2_production"]
        )
    
    # Calculate ROI and payback period (simplified)
    roi_result = {}
    if "investment" in p and "h2_selling_price" in p:
        annual_revenue = h2_result["h2_production"] * 8760 * p["h2_selling_price"]  # Assuming 24/7 operation
        if lcoh_result and "value" in lcoh_result:
            annual_cost = h2_result["h2_production"] * 8760 * lcoh_result["value"]
            annual_profit = annual_revenue - annual_cost
            if p["investment"] > 0:
                roi = (annual_profit / p["investment"]) * 100
                payback_years = p["investment"] / annual_profit if annual_profit > 0 else float('inf')
                roi_result = {
                    "annual_revenue": annual_revenue,
                    "annual_cost": annual_cost,
                    "annual_profit": annual_profit,
                    "roi_percent": roi,
                    "payback_years": payback_years
                }
    
    # Combine all results
    return {
        "hydrogen_production": h2_result,
        "economics": lcoh_result,
        "emissions": emissions_result,
        "financial": roi_result,
        "input_parameters": p
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
