"""
Qwen2-VL-7B-Instruct RunPod Handler

Generic vision model wrapper. Receives images + prompt, returns model output.
All business logic (prompts, criteria) is controlled by the caller.
"""

import runpod
import torch
import base64
import requests
import json
import re
from io import BytesIO
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = None
processor = None

def load_model():
    """Load Qwen2-VL-7B-Instruct model at startup"""
    global model, processor

    print("Loading Qwen2-VL-7B-Instruct model...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        low_cpu_mem_usage=True
    )
    model.config.torch_dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    print("Model loaded successfully!")

def download_image(url):
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image"""
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def handler(job):
    """
    RunPod handler function

    Input:
        photoUrls: list of image URLs (optional)
        photoBase64: list of base64 encoded images (optional)
        prompt: the prompt to send to the model (required)

    Output:
        success: bool
        output: raw model output text
        parsed: parsed JSON if model returned valid JSON (optional)
    """
    global model, processor

    job_input = job.get("input", {})

    photo_urls = job_input.get("photoUrls", [])
    photo_base64_list = job_input.get("photoBase64", [])
    prompt = job_input.get("prompt")

    # Validate input
    if not prompt:
        return {"success": False, "error": "prompt is required"}

    if not photo_urls and not photo_base64_list:
        return {"success": False, "error": "Either photoUrls or photoBase64 is required"}

    # Load images
    images = []
    if photo_base64_list:
        for b64 in photo_base64_list:
            img = decode_base64_image(b64)
            if img:
                images.append(img)
    else:
        for url in photo_urls:
            img = download_image(url)
            if img:
                images.append(img)

    if len(images) == 0:
        return {"success": False, "error": "No images could be loaded"}

    # Build message with images and prompt
    # Images first, then the caller's prompt as the instruction
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "user", "content": content}
    ]

    import gc

    try:
        # Log VRAM before inference
        alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
        reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"VRAM before inference: {alloc_mb:.0f}MB allocated, {reserved_mb:.0f}MB reserved")

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"Model output: {output_text}")

        # Try to parse JSON if present
        result = {"success": True, "output": output_text}

        json_match = re.search(r'\{[\s\S]*\}', output_text)
        if json_match:
            try:
                result["parsed"] = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass  # No valid JSON, just return raw output

        return result

    except RuntimeError as e:
        error_str = str(e)
        print(f"RuntimeError during inference: {error_str}")
        # CUDA errors leave GPU in unrecoverable state — let worker crash
        # so RunPod restarts it with a clean CUDA context
        if "CUDA" in error_str or "cuda" in error_str or "device-side assert" in error_str:
            print("Fatal CUDA error — letting worker crash for clean restart")
            raise
        return {"success": False, "error": error_str}

    except Exception as e:
        print(f"Error during inference: {e}")
        return {"success": False, "error": str(e)}

    finally:
        # Clean up VRAM between jobs to prevent memory creep
        torch.cuda.empty_cache()
        gc.collect()

load_model()
runpod.serverless.start({"handler": handler})
