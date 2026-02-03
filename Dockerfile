# Qwen2-VL-7B-Instruct RunPod Serverless Docker Image
# For BigTooth brushing session verification
# Model downloads at first run (cold start), then cached on volume

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python dependencies only (model downloads at runtime)
RUN pip install --no-cache-dir \
    runpod \
    transformers>=4.45.0 \
    accelerate>=0.26.0 \
    pillow \
    requests \
    qwen-vl-utils

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables - cache model on RunPod volume
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HF_HOME=/runpod-volume/huggingface

# Start handler
CMD ["python", "-u", "/app/handler.py"]
