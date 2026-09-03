"""
run_api.py
==========
CLI launcher script to run the FastAPI application server via Uvicorn.

Usage
-----
    python run_api.py
    python run_api.py --port 8000 --host 0.0.0.0
"""
import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run FastAPI Forecasting & Explainability API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"🚀 Starting FastAPI Server on http://{args.host}:{args.port}")
    print(f"📖 OpenAPI Swagger UI available at http://{args.host}:{args.port}/docs")
    print(f"📊 Timeseries API: http://{args.host}:{args.port}/api/v1/timeseries?timeframe=3months")
    print(f"🔍 Explainability API: http://{args.host}:{args.port}/api/v1/explainability")

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
