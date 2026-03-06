#!/bin/bash
echo "Starting AI Security Engine..."
# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Activate venv if it exists, otherwise just try global python
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 app.py
