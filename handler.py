"""
Qwen2-VL-7B-Instruct RunPod Handler for BigTooth Session Verification

Analyzes 4 brushing session photos to verify legitimate brushing activity.
Returns JSON with approval decision and detailed analysis.
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

# Global model and processor (loaded once at startup)
model = None
processor = None

def load_model():
    """Load Qwen2-VL-7B-Instruct model at startup"""
    global model, processor

    print("Loading Qwen2-VL-7B-Instruct model...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load with better error handling and compatibility
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Use eager attention (flash_attention_2 requires separate install)
        low_cpu_mem_usage=True
    )

    # Disable autocast if it causes issues
    model.config.torch_dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    print("Model loaded successfully!")
    print(f"Model device: {model.device}")

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
        # Remove data URI prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def build_verification_prompt(strict_mode=False):
    """Build the verification prompt based on mode"""

    base_prompt = """You are an AI reviewer for a toothbrushing app used by both children and adults.
You will analyze 4 photos taken during a 2-minute brushing session (one per mouth quadrant: top-left, top-right, bottom-left, bottom-right).

Your task is to determine if the person was genuinely brushing their teeth or attempting to fake the session.

IMPORTANT: This app is used by people of all ages in various environments. DO NOT reject based on age or environment alone.

Analyze these aspects FOR APPROVAL/REJECTION:
1. **Face Detected**: Is a person's face visible in the photos?
2. **Toothbrush Visible**: Is a toothbrush present in the photos?
3. **Toothbrush In Mouth**: Is the toothbrush actually IN or TOUCHING the person's mouth? (Not just held near face)
4. **Toothbrush Changed Location**: Does the toothbrush appear in DIFFERENT positions/locations across the 4 photos? This is critical - if the toothbrush is in the exact same position in all 4 photos, it's likely fake. Look for:
   - Different angles of the toothbrush
   - Different mouth positions
   - Movement between quadrants (top-left, top-right, bottom-left, bottom-right)
   - Natural variation in hand position
   - If all 4 photos show identical toothbrush positioning, this indicates someone just held the brush still for 4 photos (REJECT)

CRITICAL: If the toothbrush hasn't moved between photos, the person is NOT actually brushing!

Return your response in this EXACT JSON format (no other text):
{
  "approved": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "Brief explanation",
  "faceDetected": true or false,
  "toothbrushVisible": true or false,
  "toothbrushInMouth": true or false,
  "toothbrushChangedLocation": true or false,
  "suspiciousPatterns": ["list", "of", "issues"]
}

APPROVAL CRITERIA:

"""

    if strict_mode:
        base_prompt += """STRICT MODE (ALL 4 REQUIRED):
- APPROVE ONLY if ALL of these are true:
  1. Face visible ✓
  2. Toothbrush visible ✓
  3. Toothbrush IN mouth ✓
  4. Toothbrush changed location across photos ✓
- Confidence threshold: >0.8
- REJECT if ANY of the 4 criteria are not met

STRICT MODE means all 4 checks must pass. No exceptions."""
    else:
        base_prompt += """LENIENT MODE (3 of 4 REQUIRED):
- APPROVE if these 3 are true:
  1. Face visible ✓
  2. Toothbrush visible ✓
  3. Toothbrush IN mouth ✓
- Toothbrush movement is OPTIONAL (nice to have, but not required)
- Confidence threshold: >0.6
- REJECT if:
  - Missing toothbrush
  - No face visible
  - Toothbrush not in mouth (just held near face)
  - Obvious fraud (repeated photos, screen captures, etc.)

LENIENT MODE is forgiving - we understand sometimes the camera angle makes it hard to see movement, so we only require face + toothbrush + in mouth."""

    return base_prompt

def parse_ai_response(ai_response, strict_mode=False):
    """Parse AI response and apply approval logic"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', ai_response)
        if not json_match:
            raise Exception('No JSON found in AI response')

        parsed = json.loads(json_match.group())

        # Extract detailed analysis
        detailed_analysis = {
            'faceDetected': parsed.get('faceDetected', False),
            'toothbrushVisible': parsed.get('toothbrushVisible', False),
            'toothbrushInMouth': parsed.get('toothbrushInMouth', False),
            'toothbrushChangedLocation': parsed.get('toothbrushChangedLocation', False),
            'suspiciousPatterns': parsed.get('suspiciousPatterns', [])
        }

        # Apply approval logic based on mode
        final_approved = False
        final_reason = parsed.get('reason', 'AI analysis completed')
        confidence = parsed.get('confidence', 0.5)

        if strict_mode:
            # STRICT MODE: All 4 criteria must be met + confidence >= 0.8
            all_criteria_met = (
                detailed_analysis['faceDetected'] and
                detailed_analysis['toothbrushVisible'] and
                detailed_analysis['toothbrushInMouth'] and
                detailed_analysis['toothbrushChangedLocation']
            )

            meets_confidence = confidence >= 0.8
            final_approved = all_criteria_met and meets_confidence

            if not final_approved and all_criteria_met and not meets_confidence:
                confidence_pct = round(confidence * 100)
                final_reason = f"Strict mode: Confidence too low ({confidence_pct}% < 80%)"
            elif not final_approved:
                missing = []
                if not detailed_analysis['faceDetected']:
                    missing.append('face')
                if not detailed_analysis['toothbrushVisible']:
                    missing.append('toothbrush')
                if not detailed_analysis['toothbrushInMouth']:
                    missing.append('brush in mouth')
                if not detailed_analysis['toothbrushChangedLocation']:
                    missing.append('movement')
                final_reason = 'Strict mode: Missing ' + ', '.join(missing)
        else:
            # LENIENT MODE: Only 3 required (face + toothbrush + in mouth) + confidence >= 0.6
            required_criteria_met = (
                detailed_analysis['faceDetected'] and
                detailed_analysis['toothbrushVisible'] and
                detailed_analysis['toothbrushInMouth']
            )

            meets_confidence = confidence >= 0.6
            final_approved = required_criteria_met and meets_confidence

            if not final_approved and required_criteria_met and not meets_confidence:
                confidence_pct = round(confidence * 100)
                final_reason = f"Lenient mode: Confidence too low ({confidence_pct}% < 60%)"
            elif not final_approved:
                missing = []
                if not detailed_analysis['faceDetected']:
                    missing.append('face')
                if not detailed_analysis['toothbrushVisible']:
                    missing.append('toothbrush')
                if not detailed_analysis['toothbrushInMouth']:
                    missing.append('brush in mouth')
                final_reason = 'Lenient mode: Missing ' + ', '.join(missing)

        return {
            'success': True,
            'approved': final_approved,
            'confidence': confidence,
            'reason': final_reason,
            'detailedAnalysis': detailed_analysis
        }

    except Exception as e:
        print(f"Failed to parse AI response: {e}")

        # Fallback: Manual keyword analysis
        response_text = ai_response.lower()
        has_positive = any(word in response_text for word in ['brushing', 'toothbrush', 'teeth'])
        has_negative = any(word in response_text for word in ['fake', 'suspicious', 'not brushing'])

        return {
            'success': True,
            'approved': has_positive and not has_negative,
            'confidence': 0.5,
            'reason': 'AI response parsing failed - using keyword analysis'
        }

def handler(job):
    """RunPod handler function"""
    global model, processor

    job_input = job.get("input", {})

    # Get parameters
    photo_urls = job_input.get("photoUrls", [])
    photo_base64_list = job_input.get("photoBase64", [])
    strict_mode = job_input.get("strictMode", False)

    # Validate input - need either URLs or base64
    if not photo_urls and not photo_base64_list:
        return {
            "success": False,
            "error": "Either photoUrls or photoBase64 array is required"
        }

    # Load images
    images = []

    if photo_base64_list:
        # Use base64 images
        for i, b64 in enumerate(photo_base64_list):
            img = decode_base64_image(b64)
            if img:
                images.append(img)
            else:
                print(f"Failed to decode base64 image {i+1}")
    else:
        # Use URLs
        for i, url in enumerate(photo_urls):
            img = download_image(url)
            if img:
                images.append(img)
            else:
                print(f"Failed to download image {i+1} from {url}")

    # Validate we have enough images
    if len(images) < 4:
        return {
            "success": False,
            "approved": False,
            "confidence": 0,
            "reason": f"Insufficient photos for review (got {len(images)}, need 4)"
        }

    # Build the prompt
    system_prompt = build_verification_prompt(strict_mode)

    # Build message with images
    content = [{"type": "text", "text": "Please review these 4 brushing session photos:"}]

    for i, img in enumerate(images[:4]):
        content.append({
            "type": "image",
            "image": img
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    try:
        # Prepare inputs for Qwen2-VL
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

        # Generate response
        # Disable autocast to avoid PyTorch compatibility issues
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True
                )

        # Decode response
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

        # Parse and return result
        result = parse_ai_response(output_text, strict_mode)
        return result

    except Exception as e:
        print(f"Error during inference: {e}")
        return {
            "success": False,
            "approved": True,  # Fail-safe: approve on error
            "confidence": 0.5,
            "reason": f"AI review error: {str(e)}"
        }

# Load model at startup
load_model()

# Start RunPod handler
runpod.serverless.start({"handler": handler})
