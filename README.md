# Hydrogen Production Analyzer

![H2 Production Analyzer](https://img.shields.io/badge/ML--Powered-H₂%20Production%20Analyzer-teal)

An advanced machine learning-powered application for predicting hydrogen production through biogas reforming. This tool replaces time-consuming Aspen Plus simulations with trained ML models for quick and accurate predictions.

## Overview

The H₂ Production Analyzer provides engineers and researchers with immediate insights about hydrogen production processes using gradient boosting regression models trained on detailed Aspen Plus simulation data. The system provides:

- Hydrogen production predictions
- Economic analysis (LCOH, CapEx, OpEx)
- Process optimization
- Trend visualization
- Comparative scenario analysis

## System Architecture

- **Frontend**: React application with Material UI for a modern, responsive interface
- **Backend**: FastAPI server with ML models for predictions
- **ML Models**: Gradient Boosting Regression models trained on Aspen Plus simulation data

## Screenshots

<div style="text-align: center;">
    <p><strong>Chat Interface with Integrated Results</strong></p>
</div>

## Installation

### Prerequisites

- Node.js (v16 or newer)
- Python (3.8 or newer)
- Git

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hydrogen-production-analyzer.git
   cd hydrogen-production-analyzer
   ```

2. Set up a Python virtual environment:
   ```bash
   cd backend
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required Python packages:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy matplotlib seaborn
   pip install pytest httpx
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

### Start the Backend Server

1. Make sure your virtual environment is activated
2. From the backend directory, run:
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```
   The backend will start at http://localhost:8000

### Start the Frontend Development Server

1. From the frontend directory, run:
   ```bash
   npm start
   ```
   The frontend will start at http://localhost:3000

## Using the Application

1. Open your browser and navigate to http://localhost:3000
2. Use the chat interface to ask questions about hydrogen production
3. Try some example queries:
   - "Predict H₂ production at 800°C and 15 bar"
   - "What's the LCOH if electricity costs $0.12/kWh?"
   - "Show me the trend of temperature vs H₂ production"
   - "Optimize for maximum hydrogen yield"
   - "Compare H₂ output at temperatures: 600°C, 700°C, 800°C"
   - "What happens if I increase pressure from 10 to 20 bar?"

## Features

### 1. Hydrogen Production Prediction
- Estimate H₂ output based on temperature, pressure, and biogas flow parameters
- Get accurate predictions using ML models trained on Aspen Plus simulations

### 2. Economic Analysis
- Calculate Levelized Cost of Hydrogen (LCOH)
- Analyze CapEx and OpEx for hydrogen production facilities
- Project economics over time periods

### 3. Optimization
- Identify optimal parameters for maximum hydrogen yield
- Find cost-minimizing operating conditions
- Balance production vs economic factors

### 4. Trend Visualization
- Generate charts showing relationships between variables
- Visualize how changes in parameters affect hydrogen production
- Analyze temperature, pressure, and other effects on yield

### 5. Comparative Analysis
- Compare different operating scenarios
- Evaluate multiple parameter combinations simultaneously
- Make data-driven decisions about process conditions

## API Endpoints

The backend provides several RESTful API endpoints:

- `POST /predict_h2`: Predict hydrogen production
- `POST /predict_lcoh`: Predict levelized cost of hydrogen
- `GET /plot_trend`: Generate trend visualization
- `POST /optimize`: Find optimal parameters
- `POST /whatif`: Perform what-if analysis for parameter changes
- `POST /simulate`: Run comparative scenario simulations
- `GET /history`: Retrieve session history
- `POST /history/save`: Save a session to history

## Project Structure

```
hydrogen-production-analyzer/
├── backend/               # Python FastAPI backend
│   ├── models/            # ML model files
│   ├── predictor.py       # ML prediction logic
│   ├── optimizer.py       # Optimization logic
│   ├── server.py          # FastAPI server setup
│   └── ...
├── frontend/              # React frontend
│   ├── public/            # Static files
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── App.jsx        # Main application
│   │   └── ...
│   └── package.json       # Frontend dependencies
└── README.md              # This file
```

## Development

### Backend Testing

```bash
cd backend
pytest
```

### Frontend Testing

```bash
cd frontend
npm test
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project was developed as part of a research initiative to make hydrogen production analysis more accessible and efficient.
- Special thanks to all contributors and researchers who provided the simulation data and insights.
