import requests
import base64
from pathlib import Path

def test_health():
    """Test health endpoint"""
    response = requests.get("http://localhost:8000/health")
    print("Health check:", response.json())

def test_inference_files():
    """Test inference with file uploads"""
    url = "http://localhost:8000/infer"
    
    # Replace with your actual image paths
    person_image_path = "/home/srmist29/viton/TailorFit/media/person_images/hoodie4_0diMY6w.png"
    garment_image_path = "/home/srmist29/TryIt/25272206_55490744_600.png"
    mask_image_path = "/home/srmist29/viton/TailorFit/media/mask_images/mask_10.png"
    
    if not all(Path(p).exists() for p in [person_image_path, garment_image_path, mask_image_path]):
        print("❌ Test images not found. Please provide test images.")
        return
    
    with open(person_image_path, "rb") as f1, \
         open(garment_image_path, "rb") as f2, \
         open(mask_image_path, "rb") as f3:
        
        files = {
            "person_image": ("person.jpg", f1, "image/jpeg"),
            "garment_image": ("garment.jpg", f2, "image/jpeg"),
            "mask_image": ("mask.jpg", f3, "image/jpeg")
        }
        
        data = {
            "ddim_steps": 30,
            "scale": 7.5,
            "seed": 42,
            "height": 512,
            "width": 384
        }
        
        response = requests.post(url, files=files, data=data)
        print("Inference response:", response.json())
        
        if response.status_code == 200:
            result = response.json()
            if result["success"] and result["download_url"]:
                # Download the result
                download_response = requests.get(f"http://localhost:8000{result['download_url']}")
                if download_response.status_code == 200:
                    with open(f"downloaded_{result['result_filename']}", "wb") as f:
                        f.write(download_response.content)
                    print(f"✅ Result saved as downloaded_{result['result_filename']}")

def test_inference_base64():
    """Test inference with base64 encoded images"""
    url = "http://localhost:8000/infer_base64"
    
    # Replace with your actual image paths
    person_image_path = "test_images/person.jpg"
    garment_image_path = "test_images/garment.jpg"
    mask_image_path = "test_images/mask.jpg"
    
    if not all(Path(p).exists() for p in [person_image_path, garment_image_path, mask_image_path]):
        print("❌ Test images not found for base64 test.")
        return
    
    # Encode images to base64
    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    person_b64 = encode_image(person_image_path)
    garment_b64 = encode_image(garment_image_path)
    mask_b64 = encode_image(mask_image_path)
    
    data = {
        "person_image_b64": person_b64,
        "garment_image_b64": garment_b64,
        "mask_image_b64": mask_b64,
        "request_params": '{"ddim_steps": 30, "scale": 1.0, "seed": 42}'
    }
    
    response = requests.post(url, data=data)
    print("Base64 inference response status:", response.status_code)
    
    if response.status_code == 200:
        result = response.json()
        if result["success"] and "result_image" in result:
            # Save the base64 result
            result_bytes = base64.b64decode(result["result_image"])
            with open("base64_result.png", "wb") as f:
                f.write(result_bytes)
            print("✅ Base64 result saved as base64_result.png")

if __name__ == "__main__":
    print("Testing Virtual Try-On API...")
    #test_health()
    test_inference_files()
    # test_inference_base64()
