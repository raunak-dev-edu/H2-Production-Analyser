# Hydrogen Production Analyzer Backend

This is the FastAPI backend for the Hydrogen Production Analyzer application.

## Deployment Instructions for Render.com

1. Create a new Web Service on Render.com
2. Connect your repository
3. Use the following settings:
   - Name: hydrogen-analyzer-api
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -k uvicorn.workers.UvicornWorker server:app --timeout 60`

## Local Development

To run the server locally:

```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server:app --reload
```

The API will be available at http://localhost:8000
