"""
End-to-end test: API-1 (port 8001) → API-2 (port 8000)
"""
import requests
import json
import sys
import os

API1_URL = "http://localhost:8001/predict"
API2_URL = "http://localhost:8000/predict"
# You need to provide a valid test image path here
TEST_IMAGE = "test_image.jpg"  # Update this path to a real image file

print("=" * 60)
print("CHECKING TEST IMAGE EXISTS...")
print("=" * 60)

if not os.path.exists(TEST_IMAGE):
    print(f"❌ Test image not found: {TEST_IMAGE}")
    print("Please:")
    print("1. Place a test image (jpg/png) in the current directory")
    print("2. Update the TEST_IMAGE variable in this script")
    sys.exit(1)

print(f"✅ Test image found: {TEST_IMAGE}")

print("=" * 60)
print("STEP 1: Sending image to API-1 (Clothing Material Identifier)")
print("=" * 60)

try:
    with open(TEST_IMAGE, "rb") as f:
        response1 = requests.post(API1_URL, files={"file": ("test_shirt.png", f, "image/png")})
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API-1. Make sure it's running on port 8001")
    print("Run: python -m app.main")
    sys.exit(1)

print(f"Status: {response1.status_code}")
api1_output = response1.json()
print(f"API-1 Response:\n{json.dumps(api1_output, indent=2)}\n")

if response1.status_code != 200:
    print("API-1 failed. Stopping.")
    sys.exit(1)

print("=" * 60)
print("STEP 2: Sending API-1 output to API-2 (Washing Prediction)")
print("=" * 60)

# Verify API-1 response has required fields for API-2
required_fields = ["filename", "predicted_material", "dirt_level", "garment_type", 
                   "color_intensity", "fabric_care", "stain_types", "wash_urgency", 
                   "is_retry"]

missing_fields = [field for field in required_fields if field not in api1_output]
if missing_fields:
    print(f"⚠️  API-1 response missing fields: {missing_fields}")
    print("This might be from an older API version. Proceeding anyway...")

try:
    response2 = requests.post(API2_URL, json=api1_output)
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API-2. Make sure it's running on port 8000")
    print("Run: python main.py")
    sys.exit(1)

print(f"Status: {response2.status_code}")
api2_output = response2.json()
print(f"API-2 Response:\n{json.dumps(api2_output, indent=2)}\n")

if response2.status_code == 200:
    print("✅ END-TO-END PIPELINE SUCCESS!")
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS RESULTS:")
    print("=" * 60)
    
    # Display API-1 results
    print("📸 CLOTHING ANALYSIS (API-1):")
    print(f"   Material: {api1_output.get('predicted_material', 'N/A')}")
    print(f"   Dirt Level: {api1_output.get('dirt_level', 'N/A')}")
    print(f"   Garment Type: {api1_output.get('garment_type', 'N/A')}")
    print(f"   Color Intensity: {api1_output.get('color_intensity', 'N/A')}")
    print(f"   Fabric Care: {api1_output.get('fabric_care', 'N/A')}")
    print(f"   Stain Types: {', '.join(api1_output.get('stain_types', []))}")
    print(f"   Wash Urgency: {api1_output.get('wash_urgency', 'N/A')}")
    
    # Display API-2 results  
    print("\n🔧 WASHING PARAMETERS (API-2):")
    print(f"   Wash Cycle: {api2_output.get('wash_cycle', 'N/A')}")
    print(f"   Status: {api2_output.get('status', 'N/A')}")
    print(f"   Temperature: {api2_output.get('temperature', 'N/A')}°C")
    print(f"   Water Level: {api2_output.get('water_level', 'N/A')}L")
    print(f"   Detergent: {api2_output.get('detergent_amount', 'N/A')}mL")
    print(f"   Soak Time: {api2_output.get('soak_time', 'N/A')} min")
    print(f"   Spin Time: {api2_output.get('spin_time', 'N/A')} min")
    if 'reasoning' in api2_output:
        print(f"   Reasoning: {api2_output['reasoning']}")
    print("=" * 60)
else:
    print("❌ API-2 failed.")
