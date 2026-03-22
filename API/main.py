from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json
import re
from gemini_service import predict_material, save_feedback, get_all_feedback

app = FastAPI(title="Clothing Material Identifier API")


class FeedbackRequest(BaseModel):
    image_id: str
    predicted_material: str
    correct_material: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Clothing Material Identifier API. Use /predict to analyze images."}

@app.post("/predict")
async def predict_clothing_material(
    files: List[UploadFile] = File(..., description="Upload 1 to 5 clothing images"),
    incorrect_prediction: Optional[str] = Form(None, description="Optional: provide if a previous prediction was wrong (applies to all images).")
):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="A maximum of 5 images can be submitted at once.")

    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' must be an image.")

    results = []
    for f in files:
        error_msg = None
        parsed = {}
        try:
            image_bytes = await f.read()
            prediction = await predict_material(image_bytes, incorrect_prediction)

            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?|```", "", prediction).strip()

            try:
                parsed = json.loads(clean)
                if "error" in parsed:
                    raise HTTPException(status_code=400, detail="this is not fabric")
            except json.JSONDecodeError:
                error_msg = f"Could not parse model response: {prediction}"

        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)

        results.append({
            "filename": f.filename,
            "material_type": parsed.get("material_type") if parsed else None,
            "fiber_category": parsed.get("fiber_category") if parsed else None,
            "description": parsed.get("description") if parsed else None,
            "dirt_level": parsed.get("dirt_level") if parsed else None,
            "confidence_score": parsed.get("confidence_score") if parsed else None,
            "is_retry": incorrect_prediction is not None,
            "previous_wrong_prediction": incorrect_prediction,
            "error": error_msg,
        })

    return {
        "total_images": len(results),
        "results": results,
        "hint": "If a prediction is wrong, retry with 'incorrect_prediction' parameter or use /feedback to submit a correction.",
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
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)
