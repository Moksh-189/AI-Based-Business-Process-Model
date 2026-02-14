import requests
import json
import math

try:
    response = requests.get('http://localhost:8000/api/topology')
    print(f"Status Code: {response.status_code}")
    content = response.text
    print(f"Content Body: {content[:100]}...")
    
    # Check for NaN literal which allows python-json but breaks js-json
    if "NaN" in content:
        print("CRITICAL: Found 'NaN' in JSON response! This will break frontend.")
    elif "Infinity" in content:
        print("CRITICAL: Found 'Infinity' in JSON response! This will break frontend.")
    else:
        print("JSON seems safe from NaN/Infinity.")

    # Try to parse as strict JSON
    try:
        data = json.loads(content)
        print("Python JSON parse successful.")
        print(f"Keys: {list(data.keys())}")
        print(f"Nodes: {len(data.get('nodes', []))}")
        print(f"Edges: {len(data.get('edges', []))}")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")

except Exception as e:
    print(f"Request Failed: {e}")
