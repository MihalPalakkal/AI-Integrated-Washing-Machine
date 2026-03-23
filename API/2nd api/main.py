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

# Use same naming pattern as working API-1 (gemini-flash-latest)
GEMINI_MODELS = [
    "gemini-flash-latest",             # Works! Same as API-1
    "gemini-2.5-flash",                # Gemini 2.5 Flash (stable name)
]

# Add delay between requests to avoid rate limits
last_gemini_request = 0
RATE_LIMIT_DELAY = 2  # seconds between requests

SYSTEM_INSTRUCTION = (
    "Objective: Using the material type, fiber category, and dirt level from the clothing analysis, "
    "calculate and return exact mechanical parameters for a single-item wash cycle in a fully automatic machine.\n\n"
    
    "Input Analysis:\n"
    "- Material Type: Specific fabric name\n"
    "- Fiber Category: Natural/Synthetic/Semi-synthetic\n"
    "- Dirt Level: Scale 1-5\n\n"
    
    "Parameter Calculation Logic:\n"
    "Detergent Amount: Calculate exact ml based on fiber sensitivity and soil level (Baseline: 20ml for single items).\n"
    "Soak Time: Define exact minutes (0m for delicate synthetics; up to 30m for heavily soiled natural fibers).\n"
    "Spin Time: Define exact minutes (Short 3–5m for delicate/synthetic; High 10–12m for dense natural fibers).\n"
    "Water Level: Calculate exact Liters for a SINGLE GARMENT wash (not a full load). Be realistic.\n"
    "  IMPORTANT: These values are for washing ONE piece of clothing, not a full machine load.\n"
    "  A full machine load uses 40-60L, so a single item needs MUCH LESS water.\n"
    "  - Very lightweight fabrics (silk scarf, handkerchief, lace): 5–8L\n"
    "  - Lightweight fabrics (t-shirt, blouse, chiffon, organza): 8–10L\n"
    "  - Medium-weight fabrics (shirt, polyester jacket, nylon): 10–15L\n"
    "  - Standard fabrics (jeans, cotton pants, sweater, linen): 15–20L\n"
    "  - Heavy/dense fabrics (denim jacket, wool coat, canvas, terry towel): 20–25L\n"
    "  - Adjust +2-3L for dirt levels 4-5 (extra rinse water needed)\n"
    "  - NEVER exceed 25L for a single garment — that would be unrealistic\n"
    "  - Each fabric type MUST get a DIFFERENT water level based on its weight and absorbency\n"
    "Wash Cycles: State the total number of full cycles required (Range: 1–2).\n"
    "Temperature: Provide exact °C (Max 30°C for darks/synthetics; up to 60°C for white heavy-duty natural fibers).\n"
    "Mechanical Action: Gentle (delicates), Normal (everyday), Heavy Duty (tough stains).\n\n"
    
    "You must return ONLY a single valid JSON object in this exact format:\n"
    "{\n"
    "  \"washing_logic\": {\n"
    "    \"detergent_amount\": [X],\n"
    "    \"soak_time\": [X],\n"
    "    \"spin_time\": [X],\n"
    "    \"water_level\": [X],\n"
    "    \"wash_cycles\": [X],\n"
    "    \"temperature_setting\": [X],\n"
    "    \"mechanical_action\": \"[Gentle/Normal/Heavy Duty]\"\n"
    "  }\n"
    "}\n\n"
    
    "Rules:\n"
    "1. Respond ONLY with the JSON object, no additional text\n"
    "2. ALL numeric values must be plain integers (no units, no strings) — just the number\n"
    "3. detergent_amount = ml value as integer, soak_time = minutes as integer, spin_time = minutes as integer\n"
    "4. water_level = liters as integer, wash_cycles = integer (1 or 2), temperature_setting = °C as integer\n"
    "5. mechanical_action must be exactly: Gentle, Normal, or Heavy Duty\n"
)



class ClothingPredictionInput(BaseModel):
    """
    Schema for the JSON response from API-1
    (Clothing Material Identifier with new format).

    API-1 endpoint: POST /predict
    Returns: filename, material_type, fiber_category, description, 
             dirt_level, confidence_score, is_retry, previous_wrong_prediction, error
    """

    filename: str = Field(
        ..., description="Original filename of the uploaded image"
    )
    material_type: str = Field(
        ..., description="Specific fabric name from API-1"
    )
    fiber_category: str = Field(
        ..., description="Natural, Synthetic, or Semi-synthetic"
    )
    description: str = Field(
        ..., description="3-line property summary from API-1"
    )
    dirt_level: int = Field(
        ..., description="Soil level from 1-5"
    )
    confidence_score: float = Field(
        ..., description="Analysis confidence 0.0-1.0"
    )
    is_retry: bool = Field(
        ..., description="Whether this prediction was a retry after a wrong prediction"
    )
    previous_wrong_prediction: Optional[str] = Field(
        None, description="The previous wrong prediction if this is a retry"
    )
    error: Optional[str] = Field(
        None, description="Error message from API-1 if any"
    )


# ---------------------------------------------------------------------------
# API-1 input models — mirrors API-1's actual POST /predict response
# ---------------------------------------------------------------------------

class API1Result(BaseModel):
    """One item in API-1's results array (flat fields, no nesting)."""
    filename: str
    material_type: Optional[str] = None
    fiber_category: Optional[str] = None
    description: Optional[str] = None
    dirt_level: Optional[int] = None
    confidence_score: Optional[float] = None
    is_retry: bool = False
    previous_wrong_prediction: Optional[str] = None
    error: Optional[str] = None


class API1Response(BaseModel):
    """Full response from API-1's POST /predict endpoint."""
    total_images: int
    results: List[API1Result]
    hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class WashingLogic(BaseModel):
    """Washing logic parameters."""
    detergent_amount: int = Field(..., description="Detergent amount in ml")
    soak_time: int = Field(..., description="Soak time in minutes")
    spin_time: int = Field(..., description="Spin time in minutes")
    water_level: int = Field(..., description="Water level in liters")
    wash_cycles: int = Field(..., description="Number of wash cycles")
    temperature_setting: int = Field(..., description="Temperature in °C")
    mechanical_action: str = Field(..., description="Gentle/Normal/Heavy Duty")


class WashingPrediction(BaseModel):
    washing_logic: WashingLogic = Field(..., description="Washing parameters")


class ItemBreakdown(BaseModel):
    """Per-item result used to derive the combined machine settings."""
    filename: str
    material_type: str
    fiber_category: str
    dirt_level: int
    individual_params: Optional[WashingLogic] = None
    status: str = "success"
    error: Optional[str] = None


class BatchWashingResponse(BaseModel):
    """Response when washing multiple items together in one cycle."""
    total_clothes: int = Field(..., description="Number of clothing items")
    machine_settings: WashingLogic = Field(
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

BATCH_SYSTEM_INSTRUCTION = (
    "Objective: You are an expert laundry AI. You are given a wash load containing multiple clothing items. "
    "You have their individual AI-predicted washing parameters.\n\n"
    "Your job is to generate a SINGLE set of machine settings for the WHOLE load.\n\n"
    "CRITICAL RULES:\n"
    "1. Temperature: Use the strictest (lowest) temperature to protect all fabrics. Never exceed the lowest item's temp.\n"
    "2. Mechanical Action: Use the gentlest action required by any single item.\n"
    "3. Spin Time: Use the shortest spin time to protect delicates.\n"
    "4. Water Level (Liters): Base it on the largest item plus ~30-40% more water per additional item. DO NOT simply sum them up, as that results in too much water for small loads. For 2-4 items, the water level should realistically be around 25-35L. Max limit is 60L only for huge loads.\n"
    "5. Detergent: Calculate based on combined water level and highest required concentration.\n\n"
    "You must return ONLY a single valid JSON object in this format:\n"
    "{\n"
    "  \"machine_settings\": {\n"
    "    \"detergent_amount\": [integer],\n"
    "    \"soak_time\": [integer],\n"
    "    \"spin_time\": [integer],\n"
    "    \"water_level\": [integer],\n"
    "    \"wash_cycles\": [integer],\n"
    "    \"temperature_setting\": [integer],\n"
    "    \"mechanical_action\": \"[Gentle/Normal/Heavy Duty]\"\n"
    "  },\n"
    "  \"notes\": \"[Brief explanation of your logic]\"\n"
    "}"
)

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
            f"Item {i+1}: {item.material_type} ({item.fiber_category}), Dirt Level: {item.dirt_level}/5\n"
            f"   Required individually - Temp: {logic.temperature_setting}°C, Action: {logic.mechanical_action}, "
            f"Water: {logic.water_level}L, Detergent: {logic.detergent_amount}ml, Spin: {logic.spin_time}m, Soak: {logic.soak_time}m"
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
            logic_data = parsed["machine_settings"]
            
            combined_logic = WashingLogic(
                detergent_amount=logic_data["detergent_amount"],
                soak_time=logic_data["soak_time"],
                spin_time=logic_data["spin_time"],
                water_level=logic_data["water_level"],
                wash_cycles=logic_data["wash_cycles"],
                temperature_setting=logic_data["temperature_setting"],
                mechanical_action=logic_data["mechanical_action"],
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
) -> WashingPrediction:
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
        f"Material Type: {clothing_input.material_type}\n" 
        f"Fiber Category: {clothing_input.fiber_category}\n"
        f"Properties: {clothing_input.description}\n"
        f"Dirt Level: {clothing_input.dirt_level}/5\n"
        f"Analysis Confidence: {clothing_input.confidence_score}\n\n"
        
        f"SPECIFIC ANALYSIS REQUIRED FOR: {clothing_input.material_type}\n"
        "Consider the unique properties of this specific fabric material:\n"
        "- Heat sensitivity and maximum safe temperature\n"
        "- Chemical sensitivity for detergent amount\n" 
        "- Mechanical stress tolerance for spin and action\n"
        "- Soil absorption characteristics for soak time\n"
        "- Fabric weight/density for WATER LEVEL (this is a SINGLE garment wash: lightweight=5-10L, medium=10-15L, standard=15-20L, heavy=20-25L)\n"
        "- Moisture absorption rate (high-absorbency fabrics like cotton/terry need slightly more water)\n\n"
        
        "Generate PRECISE parameters for this EXACT fabric type - NOT generic default values.\n"
        "Different materials require DIFFERENT water levels, temperatures, and care approaches.\n"
        "CRITICAL: This is for ONE piece of clothing. Water level must be realistic (5-25L range). NEVER use 35L+ for a single garment.\n\n"
        
        "IMPORTANT: Return ONLY a valid JSON object. No markdown formatting, no code blocks, no explanation text.\n"
        "Return washing_logic JSON with material-specific optimized parameters."
    )

    logger.info(
        "🤖 Requesting AI analysis for: %s (%s), dirt=%d, confidence=%.2f",
        clothing_input.material_type,
        clothing_input.fiber_category,
        clothing_input.dirt_level,
        clothing_input.confidence_score,
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
                prediction = WashingPrediction(**parsed)
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
            "material_analyzed": clothing_input.material_type,
            "fiber_category": clothing_input.fiber_category
        }
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "washing-parameter-prediction"}


@app.post(
    "/predict",
    response_model=WashingPrediction,
    tags=["Prediction"],
    summary="Predict optimal washing parameters",
    description=(
        "Accepts the JSON response from API-1 (Clothing Material Identifier) "
        "and returns predicted washing configuration parameters via Gemini "
        "inference."
    ),
)
async def predict(request: ClothingPredictionInput) -> WashingPrediction:
    """
    Predict optimal washing parameters based on the clothing material
    identified by API-1.

    The request body must be the exact JSON response from API-1's
    POST /predict endpoint.
    """
    if request.error or (request.material_type and request.material_type.lower() in ["unknown", "not a fabric"]):
        raise HTTPException(status_code=400, detail="this is not fabric")

    logger.info(
        "Received prediction request — file=%s, material=%s, fiber=%s, dirt=%d",
        request.filename,
        request.material_type,
        request.fiber_category,
        request.dirt_level,
    )

    prediction = await predict_washing_params(request)

    logger.info(
        "Prediction complete — cycles=%d, action=%s, detergent=%d",
        prediction.washing_logic.wash_cycles,
        prediction.washing_logic.mechanical_action,
        prediction.washing_logic.detergent_amount,
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
        r for r in api1_response.results
        if r.material_type and not r.error and r.material_type.lower() not in ["unknown", "not a fabric"]
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
            filename=r.filename,
            material_type=r.material_type or "Unknown",
            fiber_category=r.fiber_category or "Unknown",
            description=r.description or "",
            dirt_level=r.dirt_level or 3,
            confidence_score=r.confidence_score or 0.5,
            is_retry=r.is_retry,
            previous_wrong_prediction=r.previous_wrong_prediction,
            error=r.error,
        )
        try:
            pred = await predict_washing_params(clothing_input)
            breakdown.append(ItemBreakdown(
                filename=r.filename,
                material_type=r.material_type or "Unknown",
                fiber_category=r.fiber_category or "Unknown",
                dirt_level=r.dirt_level or 3,
                individual_params=pred.washing_logic,
                status="success",
            ))
            successful_logics.append(pred.washing_logic)
            successful_itemsList.append(r)
            logger.info(
                "  OK %s — %s, dirt=%d → temp=%d°C, action=%s, water=%dL",
                r.filename, r.material_type, r.dirt_level or 3,
                pred.washing_logic.temperature_setting,
                pred.washing_logic.mechanical_action,
                pred.washing_logic.water_level,
            )
        except HTTPException as e:
            error_detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            breakdown.append(ItemBreakdown(
                filename=r.filename,
                material_type=r.material_type or "Unknown",
                fiber_category=r.fiber_category or "Unknown",
                dirt_level=r.dirt_level or 3,
                status="failed",
                error=error_detail,
            ))
            logger.warning("  FAIL %s: %s", r.filename, error_detail)

    if not successful_logics:
        raise HTTPException(
            status_code=503,
            detail="All clothing item predictions failed. Please check the AI service and retry.",
        )

    combined_logic, combo_notes = await predict_combined_washing_params(successful_itemsList, successful_logics)

    logger.info(
        "Batch complete — predicted=%d, failed=%d | Combined: temp=%d°C, action=%s, water=%dL",
        len(successful_logics),
        len(breakdown) - len(successful_logics),
        combined_logic.temperature_setting,
        combined_logic.mechanical_action,
        combined_logic.water_level,
    )

    return BatchWashingResponse(
        total_clothes=len(valid_items),
        machine_settings=combined_logic,
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
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
