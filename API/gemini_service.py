import os
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv
from typing import Optional, Dict, List
import json
import asyncio
import time

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=api_key)

# Store feedback for learning
feedback_store: List[Dict] = []

# Track last request time for rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 3  # seconds between requests

def get_gemini_model():
    return genai.GenerativeModel('gemini-flash-latest')

async def predict_material(image_bytes: bytes, previous_error: str = None, max_retries: int = 3) -> str:
    """
    Analyzes the image to identify clothing material.
    If previous_error is provided, it tries to correct the prediction.
    Includes automatic retry with delay for rate limits.
    """
    global last_request_time
    
    # Rate limiting - wait if needed
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < MIN_REQUEST_INTERVAL:
        wait_time = MIN_REQUEST_INTERVAL - time_since_last
        print(f"Rate limiting: waiting {wait_time:.1f}s...")
        await asyncio.sleep(wait_time)
    
    try:
        model = get_gemini_model()
        # Fallback list if the first model fails (e.g. 404 or 429)
        backup_models = ['gemini-pro-latest', 'gemini-flash-lite-latest']
        
        image = Image.open(io.BytesIO(image_bytes))

        base_prompt = """Objective: Analyze the provided image of a textile and return a structured JSON response identifying the material, its properties (including fiber category), and current soil level for washing optimization.

            0. Fabric Validation:
            First, verify if the given image actually contains a fabric, clothing material, or textile. If it does NOT, or if the material is completely unrecognizable, you MUST return exactly this JSON and nothing else:
            {
              "error": "Not a fabric"
            }

            1. Material Identification:

            Target: Identify the specific fabric name and weave (e.g., "Cotton Twill").

            Visual Markers: Analyze diagonal ribbing (wales), fiber luster (matte vs. shiny), and fold behavior (sharp vs. soft).

            2. Property Description (3-Line Limit):

            Provide a concise summary of the material's durability, wrinkle resistance, and breathability based on its structural properties.

            Fiber Category Integration: Explicitly state if the identified fabric belongs to the Natural, Synthetic, or Semi-synthetic category.

            3. Soil Level Assessment (Scale 1–5):

            Level 1: Clean/Light Use. No visible stains; minor surface lint or faint scuffs only.

            Level 2: Surface Soiled. Localized minor spots or slight discoloration.

            Level 3: Moderately Dirty. Visible sweat marks or multiple small stains.

            Level 4: Heavily Soiled. Large, deep-set stains (mud, grease) covering >25% of surface.

            Level 5: Emergency. Complete saturation with thick grime or heavy industrial oils.

        4. Required JSON Output Format (if it IS a fabric):
        {
          "material_type": "[Fabric Name]",
          "fiber_category": "[Natural/Synthetic/Semi-synthetic]",
          "description": "[3-line property summary including the fiber category]",
          "dirt_level": [Number 1-5],
          "confidence_score": [0.0-1.0]
        }

        Rules:
        - Return ONLY the JSON object, no additional text
        - If not a fabric, return exactly {"error": "Not a fabric"}
        - material_type: Specific fabric name (e.g., "Cotton Twill", "Polyester Blend")
        - fiber_category: Must be exactly one of: Natural, Synthetic, Semi-synthetic
        - description: Exactly 3 lines describing durability, wrinkle resistance, breathability
        - dirt_level: Integer from 1 to 5 based on soil assessment
        - confidence_score: Float from 0.0 to 1.0 indicating analysis confidence"""

        if previous_error:
            prompt = (
                f"{base_prompt}\n\n"
                f" IMPORTANT: '{previous_error}' was INCORRECT!\n"
                f"DO NOT answer '{previous_error}' again.\n"
                "Look more carefully at fiber characteristics and provide a DIFFERENT answer."
            )
        else:
            prompt = base_prompt

        try:
            last_request_time = time.time()
            response = model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as first_error:
            error_str = str(first_error)
            
           
            if "429" in error_str or "quota" in error_str.lower():
              
                wait_time = 20  # default wait
                if "retry in" in error_str.lower():
                    try:
                        import re
                        match = re.search(r'retry in (\d+)', error_str.lower())
                        if match:
                            wait_time = int(match.group(1)) + 2
                    except:
                        pass
                
                print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
                
               
                for backup_name in backup_models:
                    try:
                        print(f"Retrying with backup model: {backup_name}")
                        await asyncio.sleep(2)  # Small delay between attempts
                        backup_model = genai.GenerativeModel(backup_name)
                        response = backup_model.generate_content([prompt, image])
                        last_request_time = time.time()
                        return response.text.strip()
                    except Exception as e:
                        print(f"Backup model {backup_name} failed: {e}")
                        await asyncio.sleep(5)
                
                return "Rate limit exceeded. Please wait 1 minute and try again."
            
            print(f"Primary model failed: {first_error}. Trying backups...")
            for backup_name in backup_models:
                try:
                    print(f"Trying backup model: {backup_name}")
                    backup_model = genai.GenerativeModel(backup_name)
                    response = backup_model.generate_content([prompt, image])
                    last_request_time = time.time()
                    return response.text.strip()
                except Exception as e:
                    print(f"Backup model {backup_name} failed: {e}")
            raise first_error 

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def save_feedback(image_id: str, predicted: str, correct: str) -> Dict:
    """Save user feedback about incorrect predictions"""
    feedback = {
        "image_id": image_id,
        "predicted_material": predicted,
        "correct_material": correct
    }
    feedback_store.append(feedback)
    return feedback


def get_all_feedback() -> List[Dict]:
    return feedback_store
