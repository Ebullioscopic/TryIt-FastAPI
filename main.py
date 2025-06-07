import argparse
import os
import sys
import glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
import json
import uuid
import io
import base64
from pathlib import Path
from typing import Optional, List
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your model modules
from ldm.data.cp_dataset import CPDataset
from ldm.resizer import Resizer
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.deepfashions import DFPairDataset

import clip
from torchvision.transforms import Resize

# Initialize FastAPI app
app = FastAPI(
    title="Virtual Try-On API",
    description="API for virtual garment try-on using diffusion models",
    version="1.0.0",
    debug=True
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
sampler = None
device = None
config = None

# Create directories
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Pydantic models
class InferenceRequest(BaseModel):
    ddim_steps: int = 50
    scale: float = 1.0
    seed: int = 23
    height: int = 512
    width: int = 384
    eta: float = 0.0
    use_plms: bool = False
    n_samples: int = 1
    fixed_code: bool = False

class InferenceResponse(BaseModel):
    success: bool
    message: str
    result_filename: Optional[str] = None
    download_url: Optional[str] = None

# Utility functions from your original script
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def load_model_from_config(config_path, ckpt_path, verbose=False):
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model, config

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

# Image preprocessing functions
def preprocess_image_for_model(image_bytes: bytes, target_size: tuple = (512, 384)) -> torch.Tensor:
    """Preprocess uploaded image for model input"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((target_size[1], target_size[0]))
    
    transform = get_tensor(normalize=True, toTensor=True)
    return transform(image).unsqueeze(0)

def preprocess_mask(mask_bytes: bytes, target_size: tuple = (512, 384)) -> torch.Tensor:
    """Preprocess mask image"""
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask = mask.resize((target_size[1], target_size[0]))
    
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(mask).unsqueeze(0)

def save_temp_file(file_bytes: bytes, suffix: str = ".jpg") -> str:
    """Save temporary file and return path"""
    temp_filename = f"{uuid.uuid4()}{suffix}"
    temp_path = TEMP_DIR / temp_filename
    
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    return str(temp_path)

def cleanup_temp_file(filepath: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Failed to cleanup temp file {filepath}: {e}")

# Model initialization
def initialize_model(config_path: str, ckpt_path: str, gpu_id: int = 0):
    """Initialize the model and sampler"""
    global model, sampler, device, config
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    model, config = load_model_from_config(config_path, ckpt_path, verbose=True)
    sampler = DDIMSampler(model)
    
    print(f"Model initialized on device: {device}")
    return model is not None

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    # Update these paths to match your model files
    config_path = "configs/viton512.yaml"
    ckpt_path = "checkpoint/mvg.ckpt"
    
    if os.path.exists(config_path) and os.path.exists(ckpt_path):
        try:
            success = initialize_model(config_path, ckpt_path)
            if success:
                print("✅ Model loaded successfully!")
            else:
                print("❌ Failed to load model")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️  Model files not found:")
        print(f"   Config: {config_path} (exists: {os.path.exists(config_path)})")
        print(f"   Checkpoint: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Virtual Try-On API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not set",
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/infer", response_model=InferenceResponse)
async def inference(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="Person image"),
    garment_image: UploadFile = File(..., description="Garment reference image"),
    mask_image: UploadFile = File(..., description="Inpainting mask"),
    skeleton_front: Optional[UploadFile] = File(None, description="Front skeleton image"),
    skeleton_back: Optional[UploadFile] = File(None, description="Back skeleton image"),
    skeleton_pose: Optional[UploadFile] = File(None, description="Pose skeleton image"),
    controlnet_front: Optional[UploadFile] = File(None, description="ControlNet front condition"),
    controlnet_back: Optional[UploadFile] = File(None, description="ControlNet back condition"),
    warp_feat: Optional[UploadFile] = File(None, description="Warped feature image"),
    ddim_steps: int = Form(50),
    scale: float = Form(1.0),
    seed: int = Form(23),
    height: int = Form(512),
    width: int = Form(384),
    eta: float = Form(0.0),
    use_plms: bool = Form(False),
    n_samples: int = Form(1),
    fixed_code: bool = Form(False)
):
    """Main inference endpoint for virtual try-on"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server startup logs.")
    
    temp_files = []
    
    try:
        # Set seed for reproducibility
        seed_everything(seed)
        
        # Read uploaded files
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        mask_bytes = await mask_image.read()
        
        # Save temp files for cleanup
        person_path = save_temp_file(person_bytes, ".jpg")
        garment_path = save_temp_file(garment_bytes, ".jpg")
        mask_path = save_temp_file(mask_bytes, ".jpg")
        temp_files.extend([person_path, garment_path, mask_path])
        
        # Preprocess images
        target_size = (height, width)
        person_tensor = preprocess_image_for_model(person_bytes, target_size)
        garment_tensor = preprocess_image_for_model(garment_bytes, target_size)
        mask_tensor = preprocess_mask(mask_bytes, target_size)
        
        # Move to device
        person_tensor = person_tensor.to(device)
        garment_tensor = garment_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        
        if warp_feat:
            warp_feat_bytes = await warp_feat.read()
            feat_tensor = preprocess_image_for_model(warp_feat_bytes, target_size).to(device)
            temp_files.append(save_temp_file(warp_feat_bytes, ".jpg"))
        else:
            # Use person image as warped feature if not provided
            feat_tensor = person_tensor.clone()
            
        # Handle optional inputs
        skeleton_cf_tensor = None
        skeleton_cb_tensor = None
        skeleton_p_tensor = None
        controlnet_cond_f = None
        controlnet_cond_b = None
        feat_tensor = None
        
        if skeleton_front:
            skeleton_front_bytes = await skeleton_front.read()
            skeleton_cf_tensor = preprocess_image_for_model(skeleton_front_bytes, target_size).to(device)
            temp_files.append(save_temp_file(skeleton_front_bytes, ".jpg"))
        
        if skeleton_back:
            skeleton_back_bytes = await skeleton_back.read()
            skeleton_cb_tensor = preprocess_image_for_model(skeleton_back_bytes, target_size).to(device)
            temp_files.append(save_temp_file(skeleton_back_bytes, ".jpg"))
        
        if skeleton_pose:
            skeleton_pose_bytes = await skeleton_pose.read()
            skeleton_p_tensor = preprocess_image_for_model(skeleton_pose_bytes, target_size).to(device)
            temp_files.append(save_temp_file(skeleton_pose_bytes, ".jpg"))
        
        if controlnet_front:
            controlnet_front_bytes = await controlnet_front.read()
            controlnet_cond_f = preprocess_image_for_model(controlnet_front_bytes, target_size).to(device)
            temp_files.append(save_temp_file(controlnet_front_bytes, ".jpg"))
        
        if controlnet_back:
            controlnet_back_bytes = await controlnet_back.read()
            controlnet_cond_b = preprocess_image_for_model(controlnet_back_bytes, target_size).to(device)
            temp_files.append(save_temp_file(controlnet_back_bytes, ".jpg"))
        
        if warp_feat:
            warp_feat_bytes = await warp_feat.read()
            feat_tensor = preprocess_image_for_model(warp_feat_bytes, target_size).to(device)
            temp_files.append(save_temp_file(warp_feat_bytes, ".jpg"))
        
        # Initialize sampler
        if use_plms:
            current_sampler = PLMSSampler(model)
        else:
            current_sampler = DDIMSampler(model)
        
        # Perform inference
        with torch.no_grad():
            precision_scope = autocast if torch.cuda.is_available() else nullcontext
            with precision_scope("cuda" if torch.cuda.is_available() else "cpu"):
                with model.ema_scope():
                    print(f"🔍 DEBUG: person_tensor shape: {person_tensor.shape}")
                    print(f"🔍 DEBUG: garment_tensor shape: {garment_tensor.shape}")
                    print(f"🔍 DEBUG: mask_tensor shape: {mask_tensor.shape}")
                    # Prepare model inputs
                    test_model_kwargs = {}
                    test_model_kwargs['inpaint_mask'] = mask_tensor
                    test_model_kwargs['inpaint_image'] = person_tensor
                    
                    # Encode images to latent space
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    print(f"🔍 DEBUG: z_inpaint shape: {z_inpaint.shape}")
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    
                    # Resize mask to match latent dimensions
                    # test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                    #     test_model_kwargs['inpaint_mask'])
                    resized_mask = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])
                    test_model_kwargs['inpaint_mask'] = resized_mask
                    print(f"🔍 DEBUG: resized_mask shape: {resized_mask.shape}")
                    
                    warp_feat_encoded = model.encode_first_stage(feat_tensor)
                    warp_feat_encoded = model.get_first_stage_encoding(warp_feat_encoded).detach()
                    
                    # Create start code from warped features - Fix: Follow test.py pattern
                    ts = torch.full((1,), 999, device=device, dtype=torch.long)
                    start_code = model.q_sample(warp_feat_encoded, ts)
                    print(f"🔍 DEBUG: start_code shape: {start_code.shape}")
                    # Prepare conditioning
                    c = model.get_learned_conditioning(garment_tensor.to(torch.float16))
                    print(f"🔍 DEBUG: conditioning before proj_out shape: {c.shape}")
                    #c = model.proj_out(c)
                    c = model.proj_out(c)
                    print(f"🔍 DEBUG: conditioning after proj_out shape: {c.shape}")
                    
                    # Prepare unconditional conditioning
                    uc = None
                    if scale != 1.0:
                        uc = model.learnable_vector
                        uc = uc.repeat(garment_tensor.size(0), 1, 1)
                    
                    # Prepare start code
                    start_code = None
                    if fixed_code:
                        start_code = torch.randn([n_samples, 4, height // 8, width // 8], device=device)
                    
                    # Handle warped features if provided
                    if feat_tensor is not None:
                        warp_feat_encoded = model.encode_first_stage(feat_tensor)
                        warp_feat_encoded = model.get_first_stage_encoding(warp_feat_encoded).detach()
                        ts = torch.full((1,), 999, device=device, dtype=torch.long)
                        start_code = model.q_sample(warp_feat_encoded, ts)
                    
                    # Handle skeleton and controlnet inputs if available
                    down_samples = None
                    mid_samples = None
                    
                    if all(x is not None for x in [skeleton_cf_tensor, skeleton_cb_tensor, skeleton_p_tensor,
                                                   controlnet_cond_f, controlnet_cond_b]):
                        # Process skeleton data
                        ehs_cf = model.pose_model(skeleton_cf_tensor)
                        ehs_cb = model.pose_model(skeleton_cb_tensor)
                        ehs_p = model.pose_model(skeleton_p_tensor)
                        ehs_text = torch.zeros((c.shape[0], 1, 768)).to(device)
                        
                        # Prepare input for controlnet
                        # x_noisy = torch.cat(
                        #     (start_code if start_code is not None else torch.randn([n_samples, 4, height // 8, width // 8], device=device),
                        #      test_model_kwargs['inpaint_image'], 
                        #      test_model_kwargs['inpaint_mask']), dim=1)
                        
                        x_noisy = torch.cat(
                            (start_code, test_model_kwargs['inpaint_image'], test_model_kwargs['inpaint_mask']), dim=1)
                        
                        ts = torch.full((n_samples,), 999, device=device, dtype=torch.long)
                        
                        # Process with local controlnet
                        down_samples_f, mid_samples_f = model.local_controlnet(
                            x_noisy, ts,
                            encoder_hidden_states=ehs_text,
                            controlnet_cond=controlnet_cond_f,
                            ehs_c=ehs_cf,
                            ehs_p=ehs_p
                        )
                        
                        down_samples_b, mid_samples_b = model.local_controlnet(
                            x_noisy, ts,
                            encoder_hidden_states=ehs_text,
                            controlnet_cond=controlnet_cond_b,
                            ehs_c=ehs_cb,
                            ehs_p=ehs_p
                        )
                        
                        # Combine front and back samples
                        mid_samples = mid_samples_f + mid_samples_b
                        down_samples = ()
                        for ds in range(len(down_samples_f)):
                            tmp = torch.cat((down_samples_f[ds], down_samples_b[ds]), dim=1)
                            down_samples = down_samples + (tmp,)
                    
                    # Sampling
                    shape = [4, height // 8, width // 8]
                    samples_ddim, _ = current_sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=eta,
                        x_T=start_code,
                        down_samples=down_samples,
                        test_model_kwargs=test_model_kwargs
                    )
                    
                    # Decode samples
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    
                    # Apply mask blending
                    x_checked_image = x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    x_source = torch.clamp((person_tensor.cpu() + 1.0) / 2.0, min=0.0, max=1.0)
                    x_result = x_checked_image_torch * (1 - mask_tensor.cpu()) + mask_tensor.cpu() * x_source
                    
                    # Resize and save result
                    resize = transforms.Resize((height, int(height / 256 * 192)))
                    result_filename = f"result_{uuid.uuid4().hex}.png"
                    result_path = OUTPUT_DIR / result_filename
                    
                    for i, x_sample in enumerate(x_result):
                        save_x = resize(x_sample)
                        save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(save_x.astype(np.uint8))
                        img.save(result_path)
                        break  # Save only the first sample
        
        # Schedule cleanup of temporary files
        for temp_file in temp_files:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        download_url = f"/download/{result_filename}"
        
        return InferenceResponse(
            success=True,
            message="Inference completed successfully",
            result_filename=result_filename,
            download_url=download_url
        )
    
    except Exception as e:
        # Clean up temp files immediately on error
        for temp_file in temp_files:
            cleanup_temp_file(temp_file)
        
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/download/{filename}")
async def download_result(filename: str):
    """Download generated image"""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(filepath),
        media_type="image/png",
        filename=filename
    )

@app.post("/infer_base64")
async def inference_base64(
    person_image_b64: str = Form(...),
    garment_image_b64: str = Form(...),
    mask_image_b64: str = Form(...),
    skeleton_front_b64: Optional[str] = Form(None),
    skeleton_back_b64: Optional[str] = Form(None),
    skeleton_pose_b64: Optional[str] = Form(None),
    controlnet_front_b64: Optional[str] = Form(None),
    controlnet_back_b64: Optional[str] = Form(None),
    warp_feat_b64: Optional[str] = Form(None),
    request_params: str = Form(default='{"ddim_steps": 50, "scale": 1.0, "seed": 23}')
):
    """Alternative endpoint that accepts base64 encoded images"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Parse request parameters
        params = json.loads(request_params)
        
        # Decode base64 images
        person_bytes = base64.b64decode(person_image_b64)
        garment_bytes = base64.b64decode(garment_image_b64)
        mask_bytes = base64.b64decode(mask_image_b64)
        
        # Process similar to main inference endpoint
        seed_everything(params.get("seed", 23))
        
        target_size = (params.get("height", 512), params.get("width", 384))
        person_tensor = preprocess_image_for_model(person_bytes, target_size).to(device)
        garment_tensor = preprocess_image_for_model(garment_bytes, target_size).to(device)
        mask_tensor = preprocess_mask(mask_bytes, target_size).to(device)
        
        # Simplified inference (you can expand this based on your needs)
        with torch.no_grad():
            precision_scope = autocast if torch.cuda.is_available() else nullcontext
            with precision_scope("cuda" if torch.cuda.is_available() else "cpu"):
                with model.ema_scope():
                    # Basic inference without controlnet
                    test_model_kwargs = {
                        'inpaint_mask': mask_tensor,
                        'inpaint_image': person_tensor
                    }
                    
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])
                    
                    c = model.get_learned_conditioning(garment_tensor.to(torch.float16))
                    c = model.proj_out(c)
                    
                    uc = None
                    if params.get("scale", 1.0) != 1.0:
                        uc = model.learnable_vector
                        uc = uc.repeat(garment_tensor.size(0), 1, 1)
                    
                    current_sampler = DDIMSampler(model)
                    shape = [4, target_size[0] // 8, target_size[1] // 8]
                    
                    samples_ddim, _ = current_sampler.sample(
                        S=params.get("ddim_steps", 50),
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=params.get("scale", 1.0),
                        unconditional_conditioning=uc,
                        eta=params.get("eta", 0.0),
                        test_model_kwargs=test_model_kwargs
                    )
                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    
                    # Convert to base64
                    result_image = Image.fromarray((x_samples_ddim[0] * 255).astype(np.uint8))
                    buffer = io.BytesIO()
                    result_image.save(buffer, format='PNG')
                    result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "message": "Base64 inference completed",
            "result_image": result_b64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 inference failed: {str(e)}")

@app.delete("/cleanup")
async def cleanup_generated_images():
    """Clean up generated images directory"""
    try:
        count = 0
        for file_path in OUTPUT_DIR.glob("*.png"):
            os.remove(file_path)
            count += 1
        
        return {"message": f"Cleaned up {count} generated images"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/list_results")
async def list_results():
    """List all generated result images"""
    try:
        files = [f.name for f in OUTPUT_DIR.glob("*.png")]
        return {
            "count": len(files),
            "files": files[:50]  # Limit to 50 most recent
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # You can modify these settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1      # Keep at 1 for GPU models
    )
