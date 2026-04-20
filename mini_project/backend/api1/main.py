from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json
import re
from gemini_service import predict_material, save_feedback, get_all_feedback

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Clothing Material Identifier API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    image_id: str
    predicted_material: str
    correct_material: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Clothing Material Identifier API. Use /predict to analyze images."}

@app.post("/predict")
async def predict_clothing_material(
    files: List[UploadFile] = File(None, description="Upload 1 to 5 clothing images"),
    incorrect_prediction: Optional[str] = Form(None, description="Optional: provide if a previous prediction was wrong (applies to all images).")
):
    print(f"DEBUG: Received predict request. files={files}, incorrect_prediction={incorrect_prediction}")
    if not files or len(files) == 0:
        print("DEBUG: NO FILES PROVIDED")
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="A maximum of 5 images can be submitted at once.")

    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            print(f"DEBUG: Invalid content type {f.content_type} for {f.filename}")
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' must be an image.")

    results = []
    for f in files:
        error_msg = None
        parsed = {}
        try:
            image_bytes = await f.read()
            print(f"DEBUG: Read {len(image_bytes)} bytes from {f.filename}")
            prediction = await predict_material(image_bytes, incorrect_prediction)
            print(f"DEBUG: Gemini prediction: {prediction}")

            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?|```", "", prediction).strip()

            try:
                parsed = json.loads(clean)
                if "error" in parsed:
                    print("DEBUG: Gemini said this is not fabric")
                    raise HTTPException(status_code=400, detail="this is not fabric")
            except json.JSONDecodeError:
                error_msg = f"Could not parse model response: {prediction}"
                print(f"DEBUG: JSONDecodeError: {error_msg}")

        except HTTPException as he:
            raise he
        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: Exception: {error_msg}")

        results.append({
            "name": parsed.get("material_type", "Unknown") if parsed else "Unknown",
            "confidence": parsed.get("confidence_score", 0.0) if parsed else 0.0,
            "fiberCategory": parsed.get("fiber_category", "Unknown") if parsed else "Unknown",
            "dirtLevel": parsed.get("dirt_level", 1) if parsed else 1,
            "description": parsed.get("description", "") if parsed else ""
        })

    valid_fabrics = [r for r in results if r["name"] != "Unknown"]

    return {
        "fabrics": valid_fabrics,
        "recommendedCycle": "AI Analyzed" if valid_fabrics else "No fabrics detected"
    }


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    
    result = save_feedback(
        image_id=feedback.image_id,
        predicted=feedback.predicted_material,
        correct=feedback.correct_material
    )
    return {
        "message": "Feedback recorded successfully",
        "feedback": result,
        "hint": "You can now retry prediction with 'incorrect_prediction' parameter set to the wrong prediction"
    }


@app.get("/feedback")
async def list_feedback():
    """
    Get all recorded feedback about incorrect predictions.
    """
    feedbacks = get_all_feedback()
    return {
        "total_feedbacks": len(feedbacks),
        "feedbacks": feedbacks
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
