"""
Washing Parameter Prediction Service (API-2)
=============================================
Stage 2 of a two-stage intelligent laundry system pipeline.
Accepts the JSON response produced by API-1 (Clothing Material
Identifier) and predicts optimal washing parameters using Google
Gemini API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import requests
from typing import List, Optional

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel, Field



load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("washing-prediction")

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Please set it before starting the server."
    )

genai.configure(api_key=GEMINI_API_KEY)

# Firebase Configuration for real-time hardware sync
FIREBASE_BASE_URL = "https://ai-washing-machine-default-rtdb.asia-southeast1.firebasedatabase.app"

def sync_to_firebase(logic: WashingLogic):
    """
    Sync predicted washing parameters to the root of the Firebase Realtime Database.
    This ensures that the hardware (Arduino/ESP32) sees the new settings immediately.
    """
    try:
        url = f"{FIREBASE_BASE_URL}/.json"
        data = {
            "detergent_amount": logic.detergent,
            "soak_time": logic.soakTime,
            "spin_time": logic.spinTime,
            "wash_cycle": logic.washCycles,
            "water_level": logic.water,
            "temperature": logic.temperature,
            "status": False
        }
        # Use patch to only update these specific root keys
        response = requests.patch(url, json=data, timeout=5)
        if response.status_code == 200:
            logger.info("✅ Successfully synced AI parameters to Firebase root")
        else:
            logger.warning(f"⚠️ Firebase sync returned status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"❌ Failed to sync to Firebase: {e}")

# Use same naming pattern as working API-1 (gemini-flash-latest)
GEMINI_MODELS = [
    "gemini-flash-latest",             # Works! Same as API-1
    "gemini-2.5-flash",                # Gemini 2.5 Flash (stable name)
]

# Add delay between requests to avoid rate limits
last_gemini_request = 0
RATE_LIMIT_DELAY = 2  # seconds between requests

SYSTEM_INSTRUCTION = """Objective: Using the material name, fiber category, and dirt level from the clothing analysis, calculate and return exact mechanical parameters for a single-item wash cycle in a fully automatic machine.

Input Analysis:
- Material Name (name): Specific fabric name
- Fiber Category (fiberCategory): Natural/Synthetic/Semi-synthetic
- Dirt Level (dirtLevel): Scale 1-5

Parameter Calculation Logic:
Detergent Amount: Calculate exact ml based on fiber sensitivity and soil level (Baseline: 20ml for single items).
Soak Time: MUST be dynamic based on dirt level and fabric. Range from 5m for light soil up to 30m for heavy soil. NEVER output 0 min.
Spin Time: Define exact dynamic duration in minutes. MUST VARY based on fabric delicacy AND dirt level.
  - Delicate/fragile (silk, lace, sheer): 3-5 min max.
  - Lightweight/synthetic (polyester, activewear): 5-8 min.
  - Standard/sturdy (cotton, blends): 8-12 min.
  - Heavy/dense (denim, towels) or High Dirt (Level 4-5): 12-15 min.
  - NEVER use a static default like 10 minutes for everything. Calculate it dynamically.
Spin Time Options: Calculate EXACTLY 5 suggested spin durations that the user could choose from if they disagree with your main choice.
  - Center the options around your primary 'spinTime'.
  - For delicate fabrics: Use a tight spread (e.g., 2, 3, 4, 5, 6). NEVER suggest > 8 min for silk/lace.
  - For standard fabrics: Use a moderate spread (e.g., 4, 6, 8, 10, 12).
  - For heavy duty/denim: Use a wider spread (e.g., 8, 10, 12, 14, 16).
  - Return these as an array of 5 integers.
Duration: Define total wash duration in minutes.
Water Level: Calculate exact Liters for a SINGLE GARMENT wash (not a full load). Be realistic.
  IMPORTANT: These values are for washing ONE piece of clothing, not a full machine load.
  A full machine load uses 40-60L, so a single item needs MUCH LESS water.
  - Very lightweight fabrics (silk scarf, handkerchief, lace): 5–8L
  - Lightweight fabrics (t-shirt, blouse, chiffon, organza): 8–10L
  - Medium-weight fabrics (shirt, polyester jacket, nylon): 10–15L
  - Standard fabrics (jeans, cotton pants, sweater, linen): 15–20L
  - Heavy/dense fabrics (denim jacket, wool coat, canvas, terry towel): 20–25L
  - Adjust +2-3L for dirt levels 4-5 (extra rinse water needed)
  - NEVER exceed 25L for a single garment — that would be unrealistic
  - Each fabric type MUST get a DIFFERENT water level based on its weight and absorbency
Wash Cycles: State the total number of full cycles required (Range: 1–2).
Temperature: Provide exact °C (Max 30°C for darks/synthetics; up to 60°C for white heavy-duty natural fibers).
Agitation Pattern: Gentle, Normal, Heavy Duty.

Required JSON Format:
{
  "temperature": [X],
  "spinTime": [X],
  "duration": [X],
  "detergent": [X],
  "water": [X],
  "soakTime": [X],
  "washCycles": [X],
  "agitationPattern": "[Gentle/Normal/Heavy Duty]",
  "spinTimeOptions": [X, X, X, X, X]
}

Rules:
1. Respond ONLY with the JSON object, no additional text
2. ALL numeric values must be plain integers (no units, no strings) — just the number
3. detergent = ml value as integer, duration = minutes as integer, soakTime = minutes as integer
4. water = liters as integer, washCycles = integer (1 or 2), temperature = °C as integer, spinTime = minutes as integer
5. agitationPattern must be exactly: Gentle, Normal, or Heavy Duty
"""



class ClothingPredictionInput(BaseModel):
    name: str = Field(..., description="Specific fabric name from API-1")
    confidence: float = Field(..., description="Analysis confidence 0.0-1.0")
    fiberCategory: str = Field(..., description="Natural, Synthetic, or Semi-synthetic")
    dirtLevel: int = Field(..., description="Soil level from 1-5")
    description: str = Field(..., description="3-line property summary from API-1")

class API1Result(BaseModel):
    name: str
    confidence: float
    fiberCategory: str
    dirtLevel: int
    description: str

class API1Response(BaseModel):
    fabrics: List[API1Result]
    recommendedCycle: str
    total_images: int = Field(default=1)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class WashingLogic(BaseModel):
    """Washing logic parameters."""
    temperature: int = Field(..., description="Temperature in °C")
    spinTime: int = Field(..., description="Spin time in minutes")
    duration: int = Field(..., description="Duration in minutes (soak time + spin time)")
    detergent: int = Field(..., description="Detergent amount in ml")
    water: int = Field(..., description="Water level in liters")
    soakTime: int = Field(..., description="Soak time in minutes")
    washCycles: int = Field(..., description="Number of wash cycles")
    agitationPattern: str = Field(..., description="Gentle/Normal/Heavy Duty")
    spinTimeOptions: List[int] = Field(default_factory=lambda: [5, 10, 15, 20, 25], description="5 suggested spin durations for this specific fabric")


class ItemBreakdown(BaseModel):
    """Per-item result used to derive the combined machine settings."""
    name: str
    fiberCategory: str
    dirtLevel: int
    individual_params: Optional[WashingLogic] = None
    status: str = "success"
    error: Optional[str] = None


class BatchWashingResponse(BaseModel):
    """Response when washing multiple items together in one cycle."""
    total_clothes: int = Field(..., description="Number of clothing items")
    machineSettings: WashingLogic = Field(
        ..., description="Combined washing settings to dial into the machine for the whole load"
    )
    notes: str = Field(..., description="How combined parameters were calculated")
    total_predicted: int
    total_failed: int
    individual_breakdown: List[ItemBreakdown] = Field(
        ..., description="What each garment needs individually — for reference only"
    )

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Washing Parameter Prediction API",
    description=(
        "Stage 2 of the intelligent laundry pipeline. Accepts the JSON "
        "response from API-1 (Clothing Material Identifier) and returns "
        "optimized washing parameters predicted by Google Gemini."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Combine helper — AI Concentration-Based Load Scaling
# ---------------------------------------------------------------------------

BATCH_SYSTEM_INSTRUCTION = """Objective: You are an expert laundry AI. You are given a wash load containing multiple clothing items. You have their individual AI-predicted washing parameters.

Your job is to generate a SINGLE set of machine settings for the WHOLE load.

CRITICAL RULES:
1. Temperature: Use the strictest (lowest) temperature to protect all fabrics. Never exceed the lowest item's temp.
2. Agitation Pattern: Use the gentlest action required by any single item.
3. Spin Time: Define the total spin duration in minutes for the entire batch.
4. Water Level (Liters): Base it on the largest item plus ~30-40% more water per additional item. DO NOT simply sum them up, as that results in too much water for small loads. For 2-4 items, the water level should realistically be around 25-35L. Max limit is 60L only for huge loads.
5. Detergent: Calculate based on combined water level and highest required concentration.
6. Spin Time Options: Provide 5 safe spin duration chips for the WHOLE load.
   - CENTER the options around the combined 'spinTime'.
   - The options must strictly NEVER exceed the safety limit of the most delicate fabric in the batch.

Required JSON Format:
{
  "machineSettings": {
    "temperature": [integer],
    "spinTime": [integer],
    "duration": [integer],
    "detergent": [integer],
    "water": [integer],
    "soakTime": [integer],
    "washCycles": [integer],
    "agitationPattern": "[Gentle/Normal/Heavy Duty]",
    "spinTimeOptions": [integer, integer, integer, integer, integer]
  },
  "notes": "[Brief explanation of your logic]"
}"""

async def predict_combined_washing_params(valid_items: list, logics: List[WashingLogic]) -> tuple:
    """Predict combined machine settings dynamically using the Gemini API."""
    if len(logics) == 1:
        return logics[0], "Single item — no combination needed."

    global last_gemini_request
    
    current_time = time.time()
    time_since_last = current_time - last_gemini_request
    if time_since_last < RATE_LIMIT_DELAY:
        wait_time = RATE_LIMIT_DELAY - time_since_last
        await asyncio.sleep(wait_time)

    # Build context from successful item analysis
    item_summaries = []
    for i, (item, logic) in enumerate(zip(valid_items, logics)):
        summary = (
            f"Item {i+1}: {item.name} ({item.fiberCategory}), Dirt Level: {item.dirtLevel}/5\n"
            f"   Required individually - Temp: {logic.temperature}°C, Action: {logic.agitationPattern}, "
                        f"Water: {logic.water}L, Detergent: {logic.detergent}ml, Spin: {logic.spinTime}min, Soak: {logic.soakTime}m"
        )
        item_summaries.append(summary)

    user_prompt = "Combine the following wash load:\n\n" + "\n\n".join(item_summaries) + "\n\nProvide the safest combined machine settings as JSON."

    logger.info("🤖 Requesting AI combination for %d items load.", len(logics))

    last_error = None
    for attempt, model_name in enumerate(GEMINI_MODELS):
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=BATCH_SYSTEM_INSTRUCTION,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                ),
            )
            
            if attempt > 0:
                await asyncio.sleep(3)
                
            last_gemini_request = time.time()
            response = await model.generate_content_async(user_prompt)
            
            response_text = response.text.strip()
            response_text = re.sub(r"```(?:json)?", "", response_text).strip()
            if response_text.startswith("{") is False:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
                    
            parsed = json.loads(response_text)
            logic_data = parsed["machineSettings"]
            
            combined_logic = WashingLogic(
                temperature=logic_data["temperature"],
                                spinTime=logic_data["spinTime"],
                duration=logic_data["duration"],
                detergent=logic_data["detergent"],
                water=logic_data["water"],
                soakTime=logic_data["soakTime"],
                washCycles=logic_data["washCycles"],
                agitationPattern=logic_data["agitationPattern"],
                spinTimeOptions=logic_data.get("spinTimeOptions", [5, 10, 15, 20, 25]),
            )
            return combined_logic, parsed.get("notes", "Combined parameters generated by AI.")
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < len(GEMINI_MODELS) - 1:
                    await asyncio.sleep(10)
                else: break
            else:
                await asyncio.sleep(2)
            continue
            
    raise HTTPException(status_code=503, detail=f"Failed to generate combined parameters via AI: {last_error}")


# ---------------------------------------------------------------------------
# Gemini prediction (per individual item)
# ---------------------------------------------------------------------------

async def predict_washing_params(
    clothing_input: ClothingPredictionInput,
) -> WashingLogic:
    """Send the API-1 clothing prediction to Gemini and return washing params."""
    
    global last_gemini_request
    
    # Rate limiting - wait if needed
    current_time = time.time()
    time_since_last = current_time - last_gemini_request
    if time_since_last < RATE_LIMIT_DELAY:
        wait_time = RATE_LIMIT_DELAY - time_since_last
        logger.info(f"Rate limiting: waiting {wait_time:.1f}s...")
        await asyncio.sleep(wait_time)

    user_prompt = (
        f"Analyze this specific synthetic fabric and calculate UNIQUE washing parameters:\n\n"
        f"FABRIC DETAILS:\n"
        f"Material Name: {clothing_input.name}\n" 
        f"Fiber Category: {clothing_input.fiberCategory}\n"
        f"Properties: {clothing_input.description}\n"
        f"Dirt Level: {clothing_input.dirtLevel}/5\n"
        f"Analysis Confidence: {clothing_input.confidence}\n\n"
        
        f"SPECIFIC ANALYSIS REQUIRED FOR: {clothing_input.name}\n"
        "Consider the unique properties of this specific fabric material:\n"
        "- Heat sensitivity and maximum safe temperature\n"
        "- Chemical sensitivity for detergent amount\n" 
        "- Mechanical stress tolerance for spin time and action. CRITICAL: Spin time MUST be dynamic (e.g., 5 min for delicates, 12 min for heavy/dirty, NEVER default to 10 min).\n"
        "- Soil absorption characteristics for soak time\n"
        "- Fabric weight/density for WATER LEVEL (this is a SINGLE garment wash: lightweight=5-10L, medium=10-15L, standard=15-20L, heavy=20-25L)\n"
        "- Moisture absorption rate (high-absorbency fabrics like cotton/terry need slightly more water)\n\n"
        
        "Generate PRECISE parameters for this EXACT fabric type - NOT generic default values.\n"
        "Different materials require DIFFERENT water levels, temperatures, and care approaches.\n"
        "CRITICAL: This is for ONE piece of clothing. Water level must be realistic (5-25L range). NEVER use 35L+ for a single garment.\n\n"
        
        "IMPORTANT: Return ONLY a valid JSON object. No markdown formatting, no code blocks, no explanation text.\n"
        "Return a valid JSON object with detailed wash parameters directly."
    )

    logger.info(
        "🤖 Requesting AI analysis for: %s (%s), dirt=%d, confidence=%.2f",
        clothing_input.name,
        clothing_input.fiberCategory,
        clothing_input.dirtLevel,
        clothing_input.confidence,
    )

    # Try each Gemini model
    logger.info(f"🎯 Using models: {GEMINI_MODELS}")
    last_error = None
    for attempt, model_name in enumerate(GEMINI_MODELS):
        try:
            logger.info(f"🔄 Attempt {attempt + 1}/{len(GEMINI_MODELS)}: Trying Gemini model: {model_name}")
            
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=4096,
                ),
            )
            
            # Add small delay between model attempts
            if attempt > 0:
                await asyncio.sleep(3)
            
            last_gemini_request = time.time()
            response = await model.generate_content_async(user_prompt)
            
            # Parse JSON response - handle markdown code blocks (same as API-1)
            try:
                response_text = response.text.strip()
                logger.info(f"📊 Raw AI response: {response_text[:300]}")
                
                # Extract JSON from markdown code blocks if present
                response_text = re.sub(r"```(?:json)?", "", response_text).strip()
                if response_text.startswith("{") is False:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        response_text = response_text[json_start:json_end]
                
                parsed = json.loads(response_text)
                prediction = WashingLogic(**parsed)
                logger.info(f"✅ SUCCESS! AI generated parameters using {model_name}")
                return prediction
                
            except (json.JSONDecodeError, Exception) as parse_error:
                logger.warning(f"❌ Failed to parse response from {model_name}: {parse_error}")
                logger.warning(f"Raw response: {response.text[:500] if hasattr(response, 'text') else str(response)}")
                last_error = parse_error
                continue
                
        except Exception as model_error:
            error_str = str(model_error)
            logger.error(f"❌ Model {model_name} failed: {error_str}")
            last_error = model_error
            
            # For quota errors, wait longer before next attempt
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < len(GEMINI_MODELS) - 1:  # Not the last attempt
                    logger.warning(f"⏳ Quota exceeded for {model_name}, waiting 10s before trying next model...")
                    await asyncio.sleep(10)
                else:
                    logger.error("⚠️  All models hit quota limits, using fallback")
                    break
            else:
                # For other errors, shorter wait
                await asyncio.sleep(2)
            
            continue
    
    # All AI models failed - raise error since prediction requires AI
    logger.error(f"❌ All Gemini models failed (last error: {last_error})")
    raise HTTPException(
        status_code=503,
        detail={
            "error": "AI prediction service unavailable",
            "message": "All Gemini AI models are currently unavailable. Please try again later.",
            "last_error": str(last_error),
            "material_analyzed": clothing_input.name,
            "fiber_category": clothing_input.fiberCategory
        }
    )

async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "washing-parameter-prediction"}


@app.post(
    "/predict",
    response_model=WashingLogic,
    tags=["Prediction"],
    summary="Predict optimal washing parameters",
    description=(
        "Accepts the JSON response from API-1 (Clothing Material Identifier) "
        "and returns predicted washing configuration parameters via Gemini "
        "inference."
    ),
)
async def predict(request: ClothingPredictionInput) -> WashingLogic:
    """
    Predict optimal washing parameters based on the clothing material
    identified by API-1.

    The request body must be the exact JSON response from API-1's
    POST /predict endpoint.
    """
    if request.name and request.name.lower() in ["unknown", "not a fabric"]:
        raise HTTPException(status_code=400, detail="this is not fabric")

    logger.info(
        "Received prediction request — material=%s, fiber=%s, dirt=%d",
        request.name,
        request.fiberCategory,
        request.dirtLevel,
    )

    prediction = await predict_washing_params(request)

    # Sync to Firebase root for hardware immediate access
    sync_to_firebase(prediction)

    logger.info(
        "Prediction complete — cycles=%d, action=%s, detergent=%d",
        prediction.washCycles,
        prediction.agitationPattern,
        prediction.detergent,
    )

    return prediction


@app.post(
    "/predict-from-file",
    response_model=BatchWashingResponse,
    tags=["Prediction"],
    summary="Upload API-1 JSON output and get combined washing parameters (up to 5 items)",
    description=(
        "Upload the JSON file produced by API-1 (Clothing Material Identifier). "
        "The file must contain the analysis of **1 to 5 clothing images**.\n\n"
        "- **1 image** → returns washing parameters for that single garment.\n"
        "- **2–5 images** → predicts for each item, then merges into ONE combined "
        "`machine_settings` using **Concentration-Based Load Scaling** (not a simple average):\n"
        "  - Temperature: minimum across all items (strictest care label wins)\n"
        "  - Mechanical action: gentlest required by any item\n"
        "  - Spin time: minimum (most delicate item's limit)\n"
        "  - Water: largest item full + 40% per additional + 3 L buffer\n"
        "  - Detergent: max concentration (ml/L) × combined water volume\n\n"
        "The `machine_settings` field is the single set of parameters to dial into the machine."
    ),
)
async def predict_from_json_file(file: UploadFile = File(...)) -> BatchWashingResponse:
    """
    Predict washing parameters from an uploaded API-1 JSON output file (1–5 items).
    Returns individual params per item AND a single combined machine_settings.
    """
    if not (file.filename or "").endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail="File must be a JSON file with .json extension",
        )

    try:
        content = await file.read()
        json_data = json.loads(content.decode("utf-8"))
        
        # Intercept HTTPException JSON responses from API 1 correctly
        if "detail" in json_data and "total_images" not in json_data:
            detail_msg = str(json_data["detail"]).lower()
            if "not a fabric" in detail_msg or "this is not fabric" in detail_msg:
                raise HTTPException(status_code=400, detail="this is not fabric")
            raise HTTPException(status_code=400, detail=json_data["detail"])
            
        api1_response = API1Response(**json_data)
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse API-1 JSON: {str(e)}")

    if api1_response.total_images > 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many items: the file contains {api1_response.total_images} images. "
                "A maximum of 5 clothing items is allowed per request."
            ),
        )

    # Filter to items that were successfully analyzed by API-1
    valid_items = [
        r for r in api1_response.fabrics
        if r.name and r.name.lower() not in ["unknown", "not a fabric"]
    ]
    if not valid_items:
        raise HTTPException(
            status_code=400,
            detail="this is not fabric",
        )

    logger.info("File upload — %s: %d valid item(s)", file.filename, len(valid_items))

    breakdown: List[ItemBreakdown] = []
    successful_logics: List[WashingLogic] = []
    successful_itemsList: list = []

    for r in valid_items:
        clothing_input = ClothingPredictionInput(
            name=r.name,
            fiberCategory=r.fiberCategory,
            description=r.description,
            dirtLevel=r.dirtLevel,
            confidence=r.confidence,
        )
        try:
            pred = await predict_washing_params(clothing_input)
            breakdown.append(ItemBreakdown(
                name=r.name,
                fiberCategory=r.fiberCategory,
                dirtLevel=r.dirtLevel,
                individual_params=pred,
                status="success",
            ))
            successful_logics.append(pred)
            successful_itemsList.append(r)
            logger.info(
                "  OK %s, dirt=%d → temp=%d°C, action=%s, water=%dL",
                r.name, r.dirtLevel,
                pred.temperature,
                pred.agitationPattern,
                pred.water,
            )
        except HTTPException as e:
            error_detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            breakdown.append(ItemBreakdown(
                name=r.name,
                fiberCategory=r.fiberCategory,
                dirtLevel=r.dirtLevel,
                status="failed",
                error=error_detail,
            ))
            logger.warning("  FAIL %s: %s", r.name, error_detail)

    if not successful_logics:
        raise HTTPException(
            status_code=503,
            detail="All clothing item predictions failed. Please check the AI service and retry.",
        )

    combined_logic, combo_notes = await predict_combined_washing_params(successful_itemsList, successful_logics)

    # Sync combined parameters to Firebase root for hardware immediate access
    sync_to_firebase(combined_logic)

    logger.info(
        "Batch complete — predicted=%d, failed=%d | Combined: temp=%d°C, action=%s, water=%dL",
        len(successful_logics),
        len(breakdown) - len(successful_logics),
        combined_logic.temperature,
        combined_logic.agitationPattern,
        combined_logic.water,
    )

    return BatchWashingResponse(
        total_clothes=len(valid_items),
        machineSettings=combined_logic,
        notes=combo_notes,
        total_predicted=len(successful_logics),
        total_failed=len(breakdown) - len(successful_logics),
        individual_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
