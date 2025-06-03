from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
import base64
from typing import Optional
import uvicorn
from utils import MVVTONInference

# Initialize FastAPI app
app = FastAPI(
    title="MV VTON API",
    description="Virtual Try-On API using MV VTON model",
    version="1.0.0"
)

# Global model instance
model_inference = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_inference
    try:
        config_path = "path/to/your/config.yaml"  # Update with your config path
        ckpt_path = "path/to/your/checkpoint.ckpt"  # Update with your checkpoint path
        model_inference = MVVTONInference(config_path, ckpt_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "MV VTON API is running"}

@app.post("/try-on")
async def virtual_try_on(
    person_image: UploadFile = File(..., description="Person image"),
    cloth_image: UploadFile = File(..., description="Clothing image"),
    mask_image: UploadFile = File(..., description="Mask image"),
    ddim_steps: int = Form(30, description="Number of DDIM sampling steps"),
    guidance_scale: float = Form(1.0, description="Unconditional guidance scale")
):
    """
    Perform virtual try-on inference
    
    - **person_image**: Image of the person
    - **cloth_image**: Image of the clothing item
    - **mask_image**: Mask indicating where to apply the clothing
    - **ddim_steps**: Number of sampling steps (default: 30)
    - **guidance_scale**: Guidance scale for sampling (default: 1.0)
    """
    
    if model_inference is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and validate images
        person_img = Image.open(io.BytesIO(await person_image.read()))
        cloth_img = Image.open(io.BytesIO(await cloth_image.read()))
        mask_img = Image.open(io.BytesIO(await mask_image.read()))
        
        # Perform inference
        result_image = model_inference.infer(
            person_image=person_img,
            cloth_image=cloth_img,
            mask_image=mask_img,
            ddim_steps=ddim_steps,
            scale=guidance_scale
        )
        
        # Convert result to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=try_on_result.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/try-on-base64")
async def virtual_try_on_base64(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    ddim_steps: int = Form(30),
    guidance_scale: float = Form(1.0)
):
    """
    Perform virtual try-on and return result as base64 encoded image
    """
    
    if model_inference is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read images
        person_img = Image.open(io.BytesIO(await person_image.read()))
        cloth_img = Image.open(io.BytesIO(await cloth_image.read()))
        mask_img = Image.open(io.BytesIO(await mask_image.read()))
        
        # Perform inference
        result_image = model_inference.infer(
            person_image=person_img,
            cloth_image=cloth_img,
            mask_image=mask_img,
            ddim_steps=ddim_steps,
            scale=guidance_scale
        )
        
        # Convert to base64
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return {
            "result_image": img_base64,
            "message": "Virtual try-on completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
