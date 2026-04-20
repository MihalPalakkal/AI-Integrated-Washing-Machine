"""
End-to-end test: API-1 (port 8000) → API-2 (port 8002)
Updated to match current camelCase API response formats.
"""
import requests
import json
import sys
import os

API1_URL = "http://localhost:8000/predict"
API2_URL = "http://localhost:8002/predict"
# You need to provide a valid test image path here
TEST_IMAGE = "OIP.jpg"  # Using the existing test image

print("=" * 60)
print("CHECKING TEST IMAGE EXISTS...")
print("=" * 60)

if not os.path.exists(TEST_IMAGE):
    print(f"ERROR: Test image not found: {TEST_IMAGE}")
    print("Please:")
    print("1. Place a test image (jpg/png) in the current directory")
    print("2. Update the TEST_IMAGE variable in this script")
    sys.exit(1)

print(f"SUCCESS: Test image found: {TEST_IMAGE}")

print("=" * 60)
print("STEP 1: Sending image to API-1 (Clothing Material Identifier)")
print("=" * 60)

try:
    with open(TEST_IMAGE, "rb") as f:
        response1 = requests.post(API1_URL, files={"files": (TEST_IMAGE, f, "image/jpeg")})
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to API-1. Make sure it's running on port 8000")
    print("Run: cd api1 && python main.py")
    sys.exit(1)

print(f"Status: {response1.status_code}")
api1_output = response1.json()
print(f"API-1 Response:\n{json.dumps(api1_output, indent=2)}\n")

if response1.status_code != 200:
    print("API-1 failed. Stopping.")
    sys.exit(1)

# Verify API-1 response has the current format fields
fabrics = api1_output.get("fabrics", [])
if not fabrics:
    print("ERROR: API-1 returned no fabrics")
    sys.exit(1)

first_result = fabrics[0]
required_fields = ["name", "fiberCategory", "description", "dirtLevel", "confidence"]

missing_fields = [field for field in required_fields if field not in first_result]
if missing_fields:
    print(f"WARNING: API-1 result missing fields: {missing_fields}")
else:
    print("SUCCESS: API-1 response has all required fields in camelCase")

print("\n" + "=" * 60)
print("STEP 2: Sending API-1 result to API-2 (Washing Prediction)")
print("=" * 60)

try:
    response2 = requests.post(API2_URL, json=first_result)
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to API-2. Make sure it's running on port 8002")
    print("Run: cd api2 && python main.py")
    sys.exit(1)

print(f"Status: {response2.status_code}")
api2_output = response2.json()
print(f"API-2 Response:\n{json.dumps(api2_output, indent=2)}\n")

if response2.status_code == 200:
    print("SUCCESS: END-TO-END PIPELINE SUCCESS!")
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS RESULTS:")
    print("=" * 60)

    # Display API-1 results
    print("CLOTHING ANALYSIS (API-1):")
    print(f"   Material Type: {first_result.get('name', 'N/A')}")
    print(f"   Fiber Category: {first_result.get('fiberCategory', 'N/A')}")
    print(f"   Description: {first_result.get('description', 'N/A')}")
    print(f"   Dirt Level: {first_result.get('dirtLevel', 'N/A')}/5")
    print(f"   Confidence: {first_result.get('confidence', 'N/A')}")

    # Display API-2 results
    print("\nWASHING PARAMETERS (API-2):")
    print(f"   Temperature: {api2_output.get('temperature', 'N/A')} deg C")
    print(f"   Water Level: {api2_output.get('water', 'N/A')} L")
    print(f"   Detergent: {api2_output.get('detergent', 'N/A')} ml")
    print(f"   Soak Time: {api2_output.get('soakTime', 'N/A')} min")
    print(f"   Spin Time: {api2_output.get('spinTime', 'N/A')} min")
    print(f"   Duration: {api2_output.get('duration', 'N/A')} min")
    print(f"   Wash Cycles: {api2_output.get('washCycles', 'N/A')}")
    print(f"   Agitation Pattern: {api2_output.get('agitationPattern', 'N/A')}")
    print(f"   AI Spin Suggestions: {api2_output.get('spinTimeOptions', 'N/A')} min")
    print("=" * 60)
else:
    print(f"ERROR: API-2 failed with status {response2.status_code}")
else:
    print(f"❌ API-2 failed with status {response2.status_code}")
