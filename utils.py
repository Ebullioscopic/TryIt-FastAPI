import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
import torchvision
import cv2

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchvision.transforms import Resize


def load_model_from_config(config, ckpt, device, verbose=False):
    """Load the MV VTON model from config and checkpoint"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def get_tensor(normalize=True, toTensor=True):
    """Get tensor transformation for images"""
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def preprocess_image(image: Image.Image, height: int = 512, width: int = 512):
    """Preprocess uploaded image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((width, height))
    
    # Convert to tensor
    transform = get_tensor()
    tensor = transform(image).unsqueeze(0)
    
    return tensor


def postprocess_result(x_result, height: int = 512):
    """Convert model output back to PIL Image"""
    resize = transforms.Resize((height, int(height / 256 * 192)))
    
    def un_norm(x):
        return (x + 1.0) / 2.0
    
    save_x = resize(x_result)
    save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(save_x.astype(np.uint8))
    return img


class MVVTONInference:
    """MV VTON Inference class for handling model operations"""
    
    def __init__(self, config_path: str, ckpt_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        # Load configuration and model
        self.config = OmegaConf.load(config_path)
        self.model = load_model_from_config(self.config, ckpt_path, self.device)
        
        # Initialize sampler
        self.sampler = DDIMSampler(self.model)
        
        # Model parameters
        self.H = 512
        self.W = 512
        self.C = 4
        self.f = 8
        
    def infer(self, person_image: Image.Image, cloth_image: Image.Image, 
              mask_image: Image.Image, ddim_steps: int = 30, scale: float = 1.0):
        """Perform virtual try-on inference"""
        
        with torch.no_grad():
            with autocast("cuda"):
                with self.model.ema_scope():
                    # Preprocess images
                    person_tensor = preprocess_image(person_image, self.H, self.W).to(self.device)
                    cloth_tensor = preprocess_image(cloth_image, self.H, self.W).to(self.device)
                    mask_tensor = preprocess_image(mask_image, self.H, self.W).to(self.device)
                    
                    # Convert mask to single channel if needed
                    if mask_tensor.shape[1] == 3:
                        mask_tensor = mask_tensor.mean(dim=1, keepdim=True)
                    
                    # Prepare model inputs
                    test_model_kwargs = {
                        'inpaint_mask': mask_tensor,
                        'inpaint_image': person_tensor
                    }
                    
                    # Get conditioning
                    uc = None
                    if scale != 1.0:
                        uc = self.model.learnable_vector
                        uc = uc.repeat(cloth_tensor.size(0), 1, 1)
                    
                    c = self.model.get_learned_conditioning(cloth_tensor.to(torch.float16))
                    c = self.model.proj_out(c)
                    
                    # Encode inpaint image
                    z_inpaint = self.model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = self.model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])
                    
                    # Encode warped features (using cloth as reference)
                    warp_feat = self.model.encode_first_stage(cloth_tensor)
                    warp_feat = self.model.get_first_stage_encoding(warp_feat).detach()
                    
                    # Generate start code
                    ts = torch.full((1,), 999, device=self.device, dtype=torch.long)
                    start_code = self.model.q_sample(warp_feat, ts)
                    
                    # Local controlnet (if available)
                    x_noisy = torch.cat(
                        (start_code, test_model_kwargs['inpaint_image'], test_model_kwargs['inpaint_mask']), 
                        dim=1
                    )
                    
                    # Create dummy controlnet condition if not provided
                    controlnet_cond = torch.zeros((cloth_tensor.shape[0], 3, self.H, self.W)).to(self.device)
                    
                    down_samples, _ = self.model.local_controlnet(
                        x_noisy, ts, 
                        encoder_hidden_states=torch.zeros((c.shape[0], 1, 768)).to(self.device), 
                        controlnet_cond=controlnet_cond
                    )
                    
                    # Sample
                    shape = [self.C, self.H // self.f, self.W // self.f]
                    samples_ddim, _ = self.sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=0.0,
                        x_T=start_code,
                        down_samples=down_samples,
                        test_model_kwargs=test_model_kwargs
                    )
                    
                    # Decode result
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    
                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_source = torch.clamp((person_tensor + 1.0) / 2.0, min=0.0, max=1.0)
                    x_result = x_checked_image_torch * (1 - mask_tensor.cpu()) + mask_tensor.cpu() * x_source.cpu()
                    
                    # Convert to PIL Image
                    result_image = postprocess_result(x_result[0])
                    
                    return result_image
